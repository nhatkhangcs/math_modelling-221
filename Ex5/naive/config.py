class args:
    fit_path = 'fit/'
    save_fit_name = 'netfit_SP_test.pt'
    load_fit_name = 'netfit_SP_test.pt'
    view_fit_name = 'netfit_softplus_best_1.pt'
    
    abcd_path = '4/'
    save_abcd_name = 'net4_SP_test.pt'
    load_abcd_name = 'net4_SP_test.pt'
    view_abcd_name = 'net4_SP_test.pt'

    fit_lr = 0.01
    fit_num_epochs = 40000
    fit_train_portion = 0.9

    abcd_lr = 0.002
    abcd_num_epochs = 40000
    abcd_train_portion = 0.95
    
    data_path = '../data/exact.csv'
    num_samples = 1000

    figure_path = 'figures/'
    abcd_figure_name = 'abcd_1.png'