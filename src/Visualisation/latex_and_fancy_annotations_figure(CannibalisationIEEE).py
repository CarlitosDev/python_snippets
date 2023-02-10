'''
    This comes from the notebook
    /Users/carlos.aguilar/Google Drive/order/Machine Learning Part/Preparing the 3rd paper/CFAV_show_selection_for_CausalImpact.ipynb
'''


save_figure = True

# Plot one store
fig, ax = plt.subplots(2,1,figsize=(fig_w*1.5*1.5, fig_h*1.5))


plt.rcParams.update({'font.size': 24})

total_days = df_store.shape[0]
num_days = np.arange(total_days)

# get the taxonomy
sku_A = df_store.filter(regex=f'sales-{sku_id_A}').columns[0]
category_id, dept_id, _, store_name = fhelp.get_taxonomy_from_sku_name_CFAV(sku_A)

start_period = df_CI.idx_promo_days[0][0]
end_period = df_CI.idx_promo_days[0][1]

xaxis_offset = 30
start_plot = start_period - xaxis_offset
end_plot = end_period + xaxis_offset

sku_B_reg = df_CI['sku_B_regular_avg_sales'][0]
sku_B_cannibalised = df_CI['sku_B_avg_sales_during_promo_sku_A'][0]
sku_B_predicted = df_CI['avg_predicted'][0]

slot_number = df_CI['slot_number'][0]

x_axis = df_store.date

idx_store = 0
idx_axis = 0

sales_sku_A = df_store[f'sales-{sku_id_A}-{store_name}']
ax[0].plot(x_axis[start_plot:end_plot], sales_sku_A[start_plot:end_plot], label=f'{sku_id_A}-{store_name}',
        color=def_colours[idx_store], linewidth=2, alpha=0.65)

# Annotation of the cannibal's promotion 
xy_text = (x_axis[start_period+2], 2)
annotation_string = r'$\mu_{\tau}^{(i)}=E[y_{\tau}^{(i)}|P_{\tau}^{(i)}=1, A_{\tau}^{(i)}=1,C_{\tau}^{(i)}=0]$'
ax[0].annotate(annotation_string,
            xy=(x_axis[start_period-3], 10), xycoords='data',
            xytext=xy_text, textcoords='data',
            bbox=dict(boxstyle='round', fc='w'))


sales_sku_B = df_store[f'sales-{sku_id_B}-{store_name}']
ax[1].plot(x_axis[start_plot:end_plot], sales_sku_B[start_plot:end_plot], label=f'{sku_id_B}-{store_name}',
        color=def_colours[idx_store+1], linewidth=2, alpha=0.65)

ax[0].plot(x_axis.iloc[start_period:end_period+1], sales_sku_A.iloc[start_period:end_period+1],
        color=def_colours[idx_store], linewidth=3, alpha=0.95)

ax[1].plot(x_axis.iloc[start_period:end_period+1], sales_sku_B.iloc[start_period:end_period+1],
        color=def_colours[idx_store+1], linewidth=3, alpha=0.95)

ax[0].axvspan(x_axis.iloc[start_period], x_axis.iloc[end_period], alpha=0.1, color='red')
ax[1].axvspan(x_axis.iloc[start_period], x_axis.iloc[end_period], alpha=0.1, color='red')

# Work out promo A days
promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
mask_sku_A = np.zeros(total_days, dtype=bool)
mask_sku_A[start_plot:end_plot+1]=True

masked_promo_A = promo_sku_A & mask_sku_A
ax[0].plot(x_axis[masked_promo_A], sales_sku_A[masked_promo_A], 'o', label=f'Promo days', 
            color='r', linewidth=2.5, alpha=0.85)


promo_sku_B = df_store[f'promotion_flag-{sku_id_B}-{store_name}']
mask_sku_B = np.zeros(total_days, dtype=bool)
mask_sku_B[start_plot:end_plot+1]=True

masked_promo_B = promo_sku_B & mask_sku_B
ax[1].plot(x_axis[masked_promo_B], sales_sku_B[masked_promo_B], 'o', label=f'Promo days', 
            color='g', linewidth=2.5, alpha=0.85)


# Add regular days
start_period_regular = df_CI.idx_regular_days[0][0]
end_period_regular = df_CI.idx_regular_days[0][1]+1

ax[0].axvspan(x_axis.iloc[start_period_regular], x_axis.iloc[end_period_regular], alpha=0.1, color='grey')


# Plot the formula on the regular left side

arrow_mid_point = start_period_regular + round((end_period_regular-start_period_regular)/2)
xy_arrow = (x_axis[arrow_mid_point], 20)
xy_text = (x_axis[start_period_regular-6], 50)
annotation_string_reg = r'$\mu_{\tau-}^{(i)}=E[y_{\tau-}^{(i)}|P_{\tau-}^{(i)}=0, A_{\tau-}^{(i)}=1,C_{\tau-}^{(i)}=0]$'
#box_style = dict(boxstyle='round', fc='grey', alpha=0.1)
box_style = dict(boxstyle='round', fc='w')
arrow_props=dict(arrowstyle="->",connectionstyle="arc3")

ax[0].annotate(annotation_string_reg,
            xy=xy_arrow, xycoords='data',
            xytext=xy_text, textcoords='data',
            bbox=box_style, arrowprops=arrow_props)


start_post_period_regular = end_period
end_post_period_regular = end_period + 6

ax[0].axvspan(x_axis.iloc[start_post_period_regular], x_axis.iloc[end_post_period_regular], alpha=0.1, color='grey')

# Plot the formula on the regular right side
xy_text = (x_axis[start_post_period_regular-1], 50)
xy_arrow = (x_axis[end_period+3], 30)
annotation_string_reg = r'$\mu_{\tau+}^{(i)}=E[y_{\tau+}^{(i)}|P_{\tau+}^{(i)}=0, A_{\tau+}^{(i)}=1,C_{\tau+}^{(i)}=0]$'
#box_style = dict(boxstyle='round', fc='grey', alpha=0.1)
box_style = dict(boxstyle='round', fc='w')
#arrow_props=dict(arrowstyle='-[, widthB=2.0, lengthB=1.5', lw=2.0)
#arrow_props=dict(arrowstyle='-', lw=2.0)
#arrow_props=dict(facecolor='black', shrink=0.01, arrowstyle="->")
ax[0].annotate(annotation_string_reg,
            xy=xy_arrow, xycoords='data',
            xytext=xy_text, textcoords='data',
            bbox=box_style, arrowprops=arrow_props)

ax[1].axvspan(x_axis.iloc[start_period_regular], x_axis.iloc[end_period_regular], alpha=0.1, color='grey')
ax[1].axvspan(x_axis.iloc[start_post_period_regular], x_axis.iloc[end_post_period_regular], alpha=0.1, color='grey')

#ax[0].annotate('local max', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))


ax[0].legend(loc='upper left')
ax[0].set_xlabel('dates')
ax[0].set_ylabel('Cannibal')
ax[0].grid(True)
ax[0].margins(0,0.08)



# Plot the formula on the victim's regular left side
arrow_mid_point = start_period_regular + round((end_period_regular-start_period_regular)/2)
xy_arrow = (x_axis[arrow_mid_point], 25)
xy_text = (x_axis[start_period_regular-6], 1)
annotation_string_reg = r'$\nu_{\tau-}^{(k)}=E[y_{\tau-}^{(k)}|P_{\tau-}^{(k)}=0, A_{\tau-}^{(k)}=1,C_{\tau-}^{(k)}=0]$'
box_style = dict(boxstyle='round', fc='w')
arrow_props=dict(arrowstyle="->",connectionstyle="arc3")
ax[1].annotate(annotation_string_reg,
            xy=xy_arrow, xycoords='data',
            xytext=xy_text, textcoords='data',
            bbox=box_style, arrowprops=arrow_props)

# Plot the formula on the victim's regular right side
xy_text = (x_axis[start_post_period_regular-1], 2)
xy_arrow = (x_axis[end_period+3], 20)
annotation_string_reg = r'$\nu_{\tau+}^{(k)}=E[y_{\tau+}^{(k)}|P_{\tau+}^{(k)}=0, A_{\tau+}^{(k)}=1,C_{\tau+}^{(k)}=0]$'
#box_style = dict(boxstyle='round', fc='grey', alpha=0.1)
box_style = dict(boxstyle='round', fc='w')
ax[1].annotate(annotation_string_reg,
            xy=xy_arrow, xycoords='data',
            xytext=xy_text, textcoords='data',
            bbox=box_style, arrowprops=arrow_props)


# Victim's
xy_text = (x_axis[start_period+3], 50)
annotation_string = r'$\nu_{\tau}^{(k)}=E[y_{\tau}^{(k)}| A_{\tau}^{(k)}=1,C_{\tau}^{(k)}=0]$'
ax[1].annotate(annotation_string,
            xy=(x_axis[start_period-3], 10), xycoords='data',
            xytext=xy_text, textcoords='data',
            bbox=dict(boxstyle='round', fc='w'))


ax[1].legend(loc='upper left')
ax[1].set_xlabel('dates')
ax[1].set_ylabel('Victim')
ax[1].grid(True)
ax[1].margins(0,0.08)


fig.tight_layout()
if save_figure:
    folder_to_save_plots = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/Preparing the 3rd paper/examples for the paper'
    foldername_png = os.path.join(folder_to_save_plots, category_id, dept_id)
    fhelp.makeFolder(foldername_png)
    plt_filename = os.path.join(foldername_png, f'{store_name}-{sku_id_A}-{sku_id_B}-{slot_number}.png')
    plt.savefig(plt_filename, format='png')