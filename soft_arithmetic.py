import torch
import torch.nn.functional as F

# Global singleton for M and the device it currently resides on
M_global = None
M_device = None

def build_addition_matrix():
    M = torch.zeros(200, 20, dtype=torch.float32)
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
    global M_global, M_device
    if M_global is None:
        # Initialize M the first time it's requested
        M_global = build_addition_matrix().to(device)
        M_device = device
    else:
        # If M is on a different device than requested, move it
        if M_device != device:
            M_global = M_global.to(device)
            M_device = device
    return M_global

def single_digit_add_batch(M, x_dist, y_dist, c_in_dist):
    batch = x_dist.shape[0]
    combined_dist = (x_dist.unsqueeze(2).unsqueeze(3) *
                     y_dist.unsqueeze(2).unsqueeze(1) *
                     c_in_dist.unsqueeze(1).unsqueeze(1))
    combined_dist = combined_dist.view(batch, 200)

    out_dist = combined_dist @ M  # (batch,20)
    out_dist = out_dist.view(batch, 2, 10)

    d_out_dist = out_dist.sum(dim=1)  # (batch,10)
    c_out_dist = out_dist.sum(dim=2)  # (batch,2)

    # Normalize (just a safeguard)
    d_out_dist = d_out_dist / (d_out_dist.sum(dim=1, keepdim=True) + 1e-9)
    c_out_dist = c_out_dist / (c_out_dist.sum(dim=1, keepdim=True) + 1e-9)
    return d_out_dist, c_out_dist

def differentiable_addition(x_vec, y_vec, n=3):
    # Ensure M is on the correct device
    device = x_vec.device
    M = get_M_for_device(device)

    batch = x_vec.shape[0]
    x_digits = x_vec.view(batch, n, 10)
    y_digits = y_vec.view(batch, n, 10)

    c_in_dist = torch.zeros(batch, 2, dtype=x_vec.dtype, device=device)
    c_in_dist[:,0] = 1.0

    result_digits = []
    # process from right to left
    for i in reversed(range(n)):
        d_out_dist, c_out_dist = single_digit_add_batch(M, x_digits[:, i, :], y_digits[:, i, :], c_in_dist)
        result_digits.append(d_out_dist)
        c_in_dist = c_out_dist

    # handle final carry
    final_digit_dist = torch.zeros(batch, 10, dtype=x_vec.dtype, device=device)
    final_digit_dist.scatter_(1, c_in_dist.argmax(dim=1, keepdim=True), 1.0)

    result_digits.reverse()
    all_digits = [final_digit_dist] + result_digits
    out = torch.cat(all_digits, dim=1)  # (batch, (n+1)*10)
    return out

#########################################################
# Example usage (CPU or GPU)
#########################################################

if __name__ == "__main__":
    # Let's say we are on a machine with GPU.
    # If we want to run on CPU:
    device = torch.device("cpu")

    # Example: 123 + 456 = 579
    def int_to_one_hot(num_list):
        oh = []
        for d in num_list:
            vec = [0]*10
            vec[d] = 1
            oh.extend(vec)
        return torch.tensor(oh, dtype=torch.float32)

    x_vec = int_to_one_hot([1,2,3]).unsqueeze(0).to(device) # (1,30)
    y_vec = int_to_one_hot([4,5,6]).unsqueeze(0).to(device)

    out = differentiable_addition(x_vec, y_vec, n=3)
    # Convert back to digits
    def one_hot_to_digits(ohe):
        d = ohe.view(-1,10)
        return [int(torch.argmax(x)) for x in d]

    digits = one_hot_to_digits(out.squeeze(0))
    print("Result:", digits)  # Expect [0,5,7,9]
