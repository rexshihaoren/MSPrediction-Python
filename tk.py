def compare_wrappers(datasets = [], clfNames = [], opt = True, tp = 'roc_auc'):
    ''' A function that takes a list of datasets and clfNames, so that it compare the model performance (roc_auc, pr, or sd)

    	TODO: 
    		1. when meshed with Tkinter automatically display suggested save name but have the option to choose where to save
    		2.tkinter GUI design forgot the option to output heatmap
    		3. still using the notation of "classifiers1" should change to classifiers after Antoine pushes his update

    	Keyword Arguments:
    		datasets: list of all selected dataframs
    		clfNames: list of all selected clfNames
    		opt: Boolean whether use the optimizaed h5
    		tp: plot type (roc_auc, pr or sd)

    '''
    # all the combination of datasets and clfNames
    combls = ''

    for i in datasets:
        dsls += (i+'_')
    cnls = ''
    for j in clfNames:
        cnls += (j+'_')

    mean_sd_roc_auc = {}
    mean_sd_pr = {}
    mean_everything = {}
    mean_everuthing1 = {}
    for clfName in models:
        # Make sure "plots/clfName" exists
        if not os.path.exists(plot_path + clfName):
            os.makedirs(plot_path + clfName)
        roc_list = []
        pr_list = []
        clf = classifiers1[clfName]
        param_dict = param_dist_dict[clfName]
        for obj in datasets:
        	comb = i+'_'+j
        	combls += comb
            y_pred, y_true, _, _, table = open_output(clfName, obj, opt)

            # out roc results and plot folds
            mean_fpr, mean_tpr, mean_auc = plot_roc(y_pred, y_true, clfName, obj, opt, save_sub = False)
            mean_everything[comb] = [mean_fpr, mean_tpr, mean_auc]
            # out pr results and plot folds
            mean_rec, mean_prec, mean_auc1 = plot_pr(y_pred, y_true, clfName, obj, opt, save_sub = False)
            mean_everything1[comb] = [mean_rec, mean_prec, mean_auc1]

            # sd list
            roc_list0 = compare_obj_sd(clfName, obj, y_pred, y_true, table, metric = 'roc_auc', opt= opt)
            pr_list0 = compare_obj_sd(clfName, obj, y_pred, y_true, table, metric = 'pr', opt = opt)
            roc_list.append(roc_list0)
            pr_list.append(pr_list0)
        # store sd score of all roc_auc of all clfs
        mean_sd_roc_auc[clfName] = roc_list
        # store sd score of all prs of all clfs
        mean_sd_pr[clfName] = pr_list

    # initialize fig and save_path
    fig = pl.figure(figsize=(4,4),dpi=100)
    save_path = ''
    if tp == 'sd':
	    fig, save_path = plot_sd(mean_sd_roc_auc, datasets, 'roc_auc', opt)
	elif tp == 'roc_auc':
        # Compare mean roc score of all datasets with clf
        for comb in  combls:
            [mean_fpr, mean_tpr, mean_auc] = mean_everything[comb]
            pl.plot(mean_fpr, mean_tpr, lw=3, label = comb + ' (area = %0.2f)' %mean_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate',fontsize=10)
        pl.ylabel('True Positive Rate',fontsize=10)
        pl.title('Receiver Operating Characteristic',fontsize=10)
        pl.legend(loc='lower right', prop = {'size':7.5})
        pl.tight_layout()
        if opt:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ combls + 'roc_auc' +'_opt.pdf'
        else:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ combls + 'roc_auc' +'_noopt.pdf'
    elif tp == 'pr':
        # Compare pr score of all clfs
        for comb in  combls:
            [mean_rec, mean_prec, mean_auc1] = mean_everything1[comb]
            pl.plot(mean_rec, mean_prec, lw=3, label = comb + ' (area = %0.2f)' %mean_auc1)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('Recall',fontsize=30)
        pl.ylabel('Precision',fontsize=30)
        pl.title('Precision-Recall',fontsize=25)
        pl.legend(loc='lower right')
        pl.tight_layout()
        if opt:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ combls + 'pr' +'_opt.pdf'
        else:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ combls + 'pr' +'_noopt.pdf'

    else:
    	print("NO SUCH PLOTING OPTION: "+ tp)
    return fig, save_path
        