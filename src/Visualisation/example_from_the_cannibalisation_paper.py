# Use v2.0 colour cycle
def_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Fig sizes
fig_h = 10
fig_w = 18

f, ax = plt.subplots(2,1,figsize=(fig_w*1.5*1.5, fig_h*1.5))
plt.rcParams.update({'font.size': 25})

ax[0].plot(x_axis, sales_sku_A, label=f'Sales CN {sku_id_A}',
        color=def_colours[idx_store], linewidth=3, alpha=0.65)

sales_sku_B = df_store[f'sales-{sku_id_B}-{store_name}']
ax[0].plot(x_axis, sales_sku_B, label=f'Sales VC {sku_id_B} (reg={sku_B_reg:3.2f})',
        color=def_colours[idx_store+1], linewidth=3, alpha=0.65)

ax[0].plot(x_axis.iloc[start_period:end_period], sales_sku_A.iloc[start_period:end_period],
        color=def_colours[idx_store], linewidth=4, alpha=0.95)

ax[0].plot(x_axis.iloc[start_period:end_period], sales_sku_B.iloc[start_period:end_period],
        color=def_colours[idx_store+1], linewidth=4, alpha=0.95)

ax[0].axvspan(x_axis.iloc[start_period], x_axis.iloc[end_period], alpha=0.1, color='red')

promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
ax[0].plot(x_axis[promo_sku_A], sales_sku_A[promo_sku_A], '.', label=f'Promo days {sku_id_A} (can={sku_B_cannibalised:3.2f}, pred={sku_B_predicted:3.2f})', 
            color='g', linewidth=3.5, alpha=0.85)


promo_sku_B = df_store[f'promotion_flag-{sku_id_B}-{store_name}']
ax[0].plot(x_axis[promo_sku_B], sales_sku_B[promo_sku_B], 'o', label=f'Promo days {sku_id_B}', 
            color=def_colours[-4], linewidth=5.5, alpha=0.95)

ax[0].legend()
ax[0].set_xlabel('dates')
ax[0].set_ylabel(f'Analysis {store_alias}')
ax[0].grid(True)
ax[0].margins(0,0.05)



# Add the exogenous data
present_var = 'total_units_trend' in df_store.columns.tolist()
if use_trend_total_sales & present_var:
    total_units_signal = df_store['total_units_trend']
    ax[1].plot(x_axis, df_store['total_units_trend'], label=f'Trend total sales CN {dept_id}',
            color=def_colours[idx_store], linewidth=2, alpha=0.85)
    ax[1].plot(x_axis, df_store['total_units'], color=def_colours[idx_store], linewidth=1, alpha=0.35)
else:
    total_units_signal = df_store['total_units']
    ax[1].plot(x_axis, total_units_signal, label=f'Total sales {dept_id}',
            color=def_colours[idx_store], linewidth=2, alpha=0.85)



ax2 = ax[1].twinx()

ax2.plot(x_axis, df_store['T2M_MAX_adj'], label=f'Avg day temperature (C)',
        color='g', linewidth=2, alpha=0.45)


lines, labels = ax[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

ax[1].set_xlabel('dates')
ax[1].set_ylabel('Exogenous variables')
ax[1].grid(True)
ax[1].margins(0,0)
ax2.margins(0,0)

plt.tight_layout()
if save_to_file:
    print(category_id, dept_id)
    foldername_png = os.path.join(folder_to_save_plots, category_id, dept_id, 'causal_plots', store_name)
    fhelp.makeFolder(foldername_png)
    plt_filename = os.path.join(foldername_png, f'{sku_id_A}-{sku_id_B}-{slot_number}.pdf')
    plt.savefig(plt_filename)
    plt.close()
    print(f'File saved to {plt_filename}')
else:
    plt.show()