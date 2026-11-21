Main model `tfidf_lr` vs baseline `tfidf_lr`, seed 0, test set.

### 15 weakest intents for `tfidf_lr`

| intent | support | tfidf_lr F1 |
|---|---|---|
| balance_not_updated_after_bank_transfer | 40 | 0.747 |
| top_up_failed | 40 | 0.759 |
| pending_transfer | 40 | 0.810 |
| card_not_working | 40 | 0.813 |
| top_up_reverted | 40 | 0.825 |
| failed_transfer | 40 | 0.837 |
| transfer_not_received_by_recipient | 40 | 0.843 |
| wrong_exchange_rate_for_cash_withdrawal | 40 | 0.843 |
| verify_my_identity | 40 | 0.846 |
| unable_to_verify_identity | 40 | 0.850 |
| card_arrival | 40 | 0.864 |
| pending_top_up | 40 | 0.864 |
| why_verify_identity | 40 | 0.864 |
| card_payment_not_recognised | 40 | 0.872 |
| virtual_card_not_working | 40 | 0.873 |

### By query length (words)

| words | n | tfidf_lr acc | tfidf_lr macro-F1 |
|---|---|---|---|
| 1-5 | 278 | 0.903 | 0.775 |
| 6-10 | 1643 | 0.907 | 0.903 |
| 11-15 | 753 | 0.923 | 0.901 |
| 16-25 | 266 | 0.914 | 0.841 |
| >25 | 140 | 0.929 | 0.858 |

### Top confusions: `tfidf_lr`

| true | predicted | count |
|---|---|---|
| top_up_failed | top_up_reverted | 5 |
| balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient | 5 |
| card_payment_wrong_exchange_rate | wrong_exchange_rate_for_cash_withdrawal | 4 |
| verify_my_identity | why_verify_identity | 4 |
| card_arrival | card_delivery_estimate | 3 |
| fiat_currency_support | exchange_via_app | 3 |
| pin_blocked | card_not_working | 3 |
| pending_top_up | top_up_failed | 3 |
| card_acceptance | card_not_working | 3 |
| top_up_reverted | top_up_failed | 3 |

### Random misclassified test queries (`tfidf_lr`)

| text | true | predicted | conf |
|---|---|---|---|
| I'm starting to think my card is lost because it still hasn't arrived, can you help? | card_arrival | lost_or_stolen_card | 0.42 |
| Why am I being charged more ? | card_payment_wrong_exchange_rate | transaction_charged_twice | 0.37 |
| Why am I unable to take money from my card but the statement says the transaction is still pending? | pending_cash_withdrawal | pending_transfer | 0.54 |
| My card was taken from me | lost_or_stolen_card | cash_withdrawal_not_recognised | 0.30 |
| I don't want the transaction to go through now | cancel_transfer | transfer_not_received_by_recipient | 0.52 |
| how long do money transfers take? | transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 0.59 |
| Can you explain the virtual cards limit? | disposable_card_limits | getting_virtual_card | 0.69 |
| Can I freeze my card right now? | compromised_card | lost_or_stolen_card | 0.42 |
| Why is my debit card being declined when I have money? | declined_card_payment | declined_cash_withdrawal | 0.29 |
| I was retrieving money and my card wouldn't remove. | card_swallowed | declined_cash_withdrawal | 0.15 |
| My non-physical card will not work | virtual_card_not_working | card_not_working | 0.93 |
| Why is my virtual card is being declined? | virtual_card_not_working | getting_virtual_card | 0.60 |
| I didn't get the money I transferred | balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient | 0.42 |
| What can I use to verify my identify? | verify_my_identity | why_verify_identity | 0.50 |
| Is there any way to verify who I am? | verify_my_identity | why_verify_identity | 0.63 |