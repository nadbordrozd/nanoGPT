import torch
import torch.nn.functional as F
import unittest


M_dict = {}

def build_addition_matrix(device):
    """
    Build a 200x20 matrix M that maps one-hot input (x,y,c_in) to one-hot output (d_out,c_out).

    Input indexing (x,y,c_in): x in [0..9], y in [0..9], c_in in [0..1].
    index_in = x*(10*2) + y*2 + c_in, total 200 combinations.

    Output indexing (d_out,c_out):
    sum = x+y+c_in in [0..19].
    d_out = sum % 10
    c_out = sum // 10 in [0..1].
    index_out = c_out*10 + d_out, total 20 combinations.
    """
    M = torch.zeros(200, 20, dtype=torch.float32, device=device)
    for x in range(10):
        for y in range(10):
            for c_in in range(2):
                idx_in = x*(10*2) + y*2 + c_in
                s = x + y + c_in
                d_out = s % 10
                c_out = s // 10
                idx_out = c_out*10 + d_out
                M[idx_in, idx_out] = 1.0
    return M

def get_M_for_device(device):
    if device not in M_dict:
        M_dict[device] = build_addition_matrix(device)    
    return M_dict[device]


def single_digit_add_batch(M, x_dist, y_dist, c_in_dist):
    """
    Perform a single-digit addition with carry using matrix M for a batch of inputs.
    
    Inputs:
    - x_dist: (batch,10) distribution over x digits
    - y_dist: (batch,10) distribution over y digits
    - c_in_dist: (batch,2) distribution over carry_in
    
    Output:
    - d_out_dist: (batch,10) distribution over the result digit
    - c_out_dist: (batch,2) distribution over the output carry
    """
    batch = x_dist.shape[0]

    # x_dist: (batch,10)
    # y_dist: (batch,10)
    # c_in_dist: (batch,2)
    # Outer product: x_dist ⊗ y_dist ⊗ c_in_dist => (batch,10,10,2)
    combined_dist = x_dist.unsqueeze(2).unsqueeze(3) * y_dist.unsqueeze(2).unsqueeze(1) * c_in_dist.unsqueeze(1).unsqueeze(1)
    # Now combined_dist is (batch,10,10,2)
    # Flatten to (batch,200)
    combined_dist = combined_dist.view(batch, 200)

    # out_dist = combined_dist @ M  -> (batch,20)
    out_dist = combined_dist @ M

    # Reshape to (batch,2,10)
    out_dist = out_dist.view(batch, 2, 10)
    # Summation:
    # d_out distribution = sum over carry_out dimension
    d_out_dist = out_dist.sum(dim=1)  # (batch,10)
    # c_out distribution = sum over digit dimension
    c_out_dist = out_dist.sum(dim=2)  # (batch,2)

    # Normalize (should already be normalized if inputs are one-hot, but let's be safe)
    d_out_dist = d_out_dist / (d_out_dist.sum(dim=1, keepdim=True) + 1e-9)
    c_out_dist = c_out_dist / (c_out_dist.sum(dim=1, keepdim=True) + 1e-9)

    return d_out_dist, c_out_dist

def differentiable_addition(x_vec, y_vec, n=3):
    """
    Perform n-digit addition using the matrix-based method on a batch of inputs.
    x_vec, y_vec: (batch, 10*n) one-hot for each digit in the batch.
    Returns: (batch, (n+1)*10)
    """
    M = get_M_for_device(x_vec.device)
    batch = x_vec.shape[0]
    # Reshape to (batch, n, 10)
    x_digits = x_vec.view(batch, n, 10)
    y_digits = y_vec.view(batch, n, 10)

    # Initial carry distribution: all c=0
    c_in_dist = torch.zeros(batch, 2, dtype=x_vec.dtype, device=x_vec.device)
    c_in_dist[:,0] = 1.0  # carry=0 initially

    result_digits = []
    # Process from right to left
    for i in reversed(range(n)):
        d_out_dist, c_out_dist = single_digit_add_batch(M, x_digits[:, i, :], y_digits[:, i, :], c_in_dist)
        result_digits.append(d_out_dist)
        c_in_dist = c_out_dist

    # After last digit
    # final carry -> leading digit
    final_digit_dist = torch.zeros(batch, 10, dtype=x_vec.dtype, device=x_vec.device)
    # If c_out=0 => digit=0, if c_out=1 => digit=1
    final_digit_dist.scatter_(1, c_in_dist.argmax(dim=1, keepdim=True), 1.0)

    result_digits.reverse()
    all_digits = [final_digit_dist] + result_digits
    out = torch.cat(all_digits, dim=1)  # (batch, (n+1)*10)
    return out


######################################
# Example usage and unit tests
######################################

def int_to_one_hot(num_list):
    """
    Convert a list of digits to a one-hot flattened vector.
    """
    oh = []
    for d in num_list:
        vec = [0]*10
        vec[d] = 1
        oh.extend(vec)
    return torch.tensor(oh, dtype=torch.float32)

def one_hot_to_digits(ohe):
    """
    Convert a flattened one-hot digit representation back to a digit list by argmax.
    """
    d = ohe.view(-1,10)
    return [int(torch.argmax(x)) for x in d]

class TestMatrixAdditionBatch(unittest.TestCase):
    def test_single_case(self):
        # 123+456=579, n=3
        x_vec = int_to_one_hot([1,2,3]).unsqueeze(0)  # (1,30)
        y_vec = int_to_one_hot([4,5,6]).unsqueeze(0)  # (1,30)
        out = differentiable_addition(x_vec, y_vec, n=3)
        digits = one_hot_to_digits(out.squeeze(0))
        self.assertEqual(digits, [0,5,7,9])

    def test_with_carry(self):
        # 999+1=1000, n=3
        x_vec = int_to_one_hot([9,9,9]).unsqueeze(0)  # (1,30)
        y_vec = int_to_one_hot([0,0,1]).unsqueeze(0)
        out = differentiable_addition(x_vec, y_vec, n=3)
        digits = one_hot_to_digits(out.squeeze(0))
        self.assertEqual(digits, [1,0,0,0])

    def test_batch(self):
        # Test a batch of two additions:
        #  123+456=579
        #  999+1=1000
        x_batch = torch.stack([int_to_one_hot([1,2,3]), int_to_one_hot([9,9,9])], dim=0) # (2,30)
        y_batch = torch.stack([int_to_one_hot([4,5,6]), int_to_one_hot([0,0,1])], dim=0) # (2,30)
        out = differentiable_addition(x_batch, y_batch, n=3)
        # Check first example
        digits_1 = one_hot_to_digits(out[0])
        self.assertEqual(digits_1, [0,5,7,9])
        # Check second example
        digits_2 = one_hot_to_digits(out[1])
        self.assertEqual(digits_2, [1,0,0,0])

    def test_differentiability(self):
        # Test differentiability with random distributions
        batch = 4
        n = 3
        x = torch.randn(batch, 10*n, requires_grad=True)
        y = torch.randn(batch, 10*n, requires_grad=True)

        # Convert to soft distributions
        x_dist = F.softmax(x.view(batch,n,10), dim=-1).view(batch,-1)
        y_dist = F.softmax(y.view(batch,n,10), dim=-1).view(batch,-1)

        out = differentiable_addition(x_dist, y_dist, n=n)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

if __name__ == "__main__":
    unittest.main()