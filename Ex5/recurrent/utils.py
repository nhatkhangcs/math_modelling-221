import torch


# def explicit_euler_helper(R_J_pair, a, b, c, d):

#     R_J_next = [R_J_pair[0] + a*R_J_pair[0] + b*R_J_pair[0], R_J_pair[1] + c*R_J_pair[1] + d*R_J_pair[1]] 
#     R_J_next = torch.tensor(R_J_next)
#     R_J_next.requires_grad_()
    
#     return R_J_next

def explicit_euler(RJ, abcd):
    RJ.require_grad = False
    R_new = RJ[:, 0] + (abcd[0]*RJ[:, 0] + abcd[1]*RJ[:, 1]) / 1000
    R_new = R_new.unsqueeze(1)
    J_new = RJ[:, 1] + (abcd[2]*RJ[:, 0] + abcd[3]*RJ[:, 1]) / 1000
    J_new = J_new.unsqueeze(1)

    return torch.cat([R_new, J_new], dim=1)