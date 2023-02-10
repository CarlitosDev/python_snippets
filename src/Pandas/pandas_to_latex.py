pandas_to_latex.py


'''
	Quick example on how to save a DF into a tex table

	The caption and label are in the incorrect place

'''

# overall_accuracy to tex
vars_to_keep = ['MAE', 'MSE', 'RMSE', 'meanError', 'R2', 'frc_error', 'frc_bias', 'frc_acc']
str_latex = df_frc_metrics[vars_to_keep].to_latex(index=True, float_format='{:3.2f}'.format,
                                           na_rep='', bold_rows=True,
                                          column_format='c', caption='caption here', label='label here')
str_old = 'egin{tabular}{c}'
str_new = 'egin{tabular}{'+ 'c|'*(len(vars_to_keep)+1) + '}'
print(str_latex.replace(str_old, str_new))