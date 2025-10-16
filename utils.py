def print_model_params(model):
    total_params = 0
    print("参数名称    参数形状                参数数量")
    print("-" * 50)
    for name, param in model.named_parameters():
        param_num = param.numel()
        total_params += param_num
        print(f"{name:20} {list(param.shape)!s:20} {param_num:10,}")
    print("-" * 50)
    print(f"模型总参数数量: {total_params:,}")