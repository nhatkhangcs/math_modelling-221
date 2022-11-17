import torch


def explicit_euler_helper(R_J_pair, a, b, c, d):

    R_J_next = [R_J_pair[0] + a*R_J_pair[0] + b*R_J_pair[0], R_J_pair[1] + c*R_J_pair[1] + d*R_J_pair[1]] 
    R_J_next = torch.tensor(R_J_next)
    R_J_next.requires_grad_()
    
    return R_J_next

def explicit_euler(R_and_J, abcd):
    helper = explicit_euler_helper
    a = abcd[0]
    b = abcd[1]
    c = abcd[2]
    d = abcd[3]
    RJ_pred_list = [helper(RJ_pair, a, b, c, d) for RJ_pair in R_and_J]

    RJ_pred = torch.stack(RJ_pred_list)

    return RJ_pred