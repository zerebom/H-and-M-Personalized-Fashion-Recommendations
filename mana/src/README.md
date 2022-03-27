#　使い方
- end_date, no_buy_daysを指定して、pickleで保存する
```
# train期間の1日前までで判断する
end_date = phase_date(36) 
no_buy_days = 30
save_remove_list = get_remove_item(df=transactions, end_date=phase_date(36), no_buy_days=no_buy_days)


with open(f"./remove_list_use_{no_buy_days}days_no_buy", mode='wb') as f:
    pickle.dump(save_remove_list,f)

```