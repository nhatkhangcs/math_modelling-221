class args:
    fit_path = 'fit/'
    save_fit_name = 'netfit_softplus_1.pt'
    load_fit_name = 'netfit_softplus_best_1.pt'
    view_fit_name = 'netfit_softplus_1.pt'
    
    abcd_path = '4/'
    save_abcd_name = 'net4_softplus_1.pt'
    load_abcd_name = 'net4_softplus_1.pt'
    view_abcd_name = 'net4_softplus_best_of_best.pt'

    fit_lr = 0.01
    fit_num_epochs = 40000
    fit_train_portion = 0.8

    abcd_lr = 0.002
    abcd_num_epochs = 40000
    abcd_train_portion = 0.95
    
    data_path = 'data/exact.csv'
    num_samples = 1000

    figure_path = 'figures/'
    abcd_figure_name = 'abcd.png'