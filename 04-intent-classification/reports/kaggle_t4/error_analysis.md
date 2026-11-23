Main model `distilbert` vs baseline `tfidf_lr`, seed 0, test set.

**Paired bootstrap** (macro-F1 distilbert − tfidf_lr): +0.0019, 95% CI [-0.0078, +0.0115], share of resamples with no gain 0.339.

| both right | only baseline right | only main right | both wrong |
|---|---|---|---|
| 2709 | 101 | 108 | 162 |

### 15 weakest intents for `distilbert`

| intent | support | tfidf_lr F1 | distilbert F1 |
|---|---|---|---|
| pending_transfer | 40 | 0.810 | 0.761 |
| verify_my_identity | 40 | 0.846 | 0.785 |
| card_payment_not_recognised | 40 | 0.872 | 0.790 |
| why_verify_identity | 40 | 0.864 | 0.795 |
| topping_up_by_card | 40 | 0.880 | 0.800 |
| balance_not_updated_after_bank_transfer | 40 | 0.747 | 0.816 |
| declined_transfer | 40 | 0.877 | 0.824 |
| transfer_not_received_by_recipient | 40 | 0.843 | 0.831 |
| fiat_currency_support | 40 | 0.877 | 0.846 |
| top_up_reverted | 40 | 0.825 | 0.846 |
| reverted_card_payment? | 40 | 0.881 | 0.847 |
| compromised_card | 40 | 0.900 | 0.849 |
| lost_or_stolen_card | 40 | 0.914 | 0.854 |
| direct_debit_payment_not_recognised | 40 | 0.909 | 0.861 |
| declined_card_payment | 40 | 0.918 | 0.867 |

### Intents where `tfidf_lr` beats `distilbert` (34)

| intent | tfidf_lr F1 | distilbert F1 |
|---|---|---|
| card_payment_not_recognised | 0.872 | 0.790 |
| topping_up_by_card | 0.880 | 0.800 |
| why_verify_identity | 0.864 | 0.795 |
| verify_my_identity | 0.846 | 0.785 |
| lost_or_stolen_card | 0.914 | 0.854 |
| card_payment_fee_charged | 0.937 | 0.881 |
| declined_transfer | 0.877 | 0.824 |
| declined_card_payment | 0.918 | 0.867 |
| compromised_card | 0.900 | 0.849 |
| pending_transfer | 0.810 | 0.761 |
| direct_debit_payment_not_recognised | 0.909 | 0.861 |
| supported_cards_and_currencies | 0.952 | 0.905 |
| lost_or_stolen_phone | 0.987 | 0.940 |
| card_acceptance | 0.923 | 0.880 |
| receiving_money | 0.961 | 0.923 |
| reverted_card_payment? | 0.881 | 0.847 |
| fiat_currency_support | 0.877 | 0.846 |
| transfer_timing | 0.895 | 0.867 |
| beneficiary_not_allowed | 0.961 | 0.935 |
| Refund_not_showing_up | 0.974 | 0.949 |
| declined_cash_withdrawal | 0.892 | 0.870 |
| card_delivery_estimate | 0.911 | 0.892 |
| exchange_via_app | 0.889 | 0.871 |
| top_up_by_cash_or_cheque | 0.950 | 0.935 |
| top_up_by_card_charge | 0.963 | 0.950 |
| age_limit | 0.988 | 0.976 |
| transfer_not_received_by_recipient | 0.843 | 0.831 |
| card_linking | 0.963 | 0.951 |
| country_support | 0.963 | 0.951 |
| transfer_fee_charged | 0.937 | 0.925 |
| change_pin | 0.974 | 0.964 |
| pending_card_payment | 0.914 | 0.907 |
| getting_virtual_card | 0.874 | 0.867 |
| card_swallowed | 0.949 | 0.947 |

### By query length (words)

| words | n | tfidf_lr acc | tfidf_lr macro-F1 | distilbert acc | distilbert macro-F1 |
|---|---|---|---|---|---|
| 1-5 | 278 | 0.903 | 0.775 | 0.917 | 0.816 |
| 6-10 | 1643 | 0.907 | 0.903 | 0.909 | 0.906 |
| 11-15 | 753 | 0.923 | 0.901 | 0.930 | 0.911 |
| 16-25 | 266 | 0.914 | 0.841 | 0.902 | 0.873 |
| >25 | 140 | 0.929 | 0.858 | 0.914 | 0.883 |

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

### Top confusions: `distilbert`

| true | predicted | count |
|---|---|---|
| why_verify_identity | verify_my_identity | 8 |
| card_arrival | card_delivery_estimate | 5 |
| fiat_currency_support | exchange_via_app | 5 |
| card_payment_not_recognised | direct_debit_payment_not_recognised | 5 |
| declined_transfer | declined_card_payment | 5 |
| pending_transfer | transfer_not_received_by_recipient | 5 |
| get_disposable_virtual_card | getting_virtual_card | 5 |
| balance_not_updated_after_bank_transfer | transfer_timing | 5 |
| verify_my_identity | why_verify_identity | 4 |
| verify_my_identity | unable_to_verify_identity | 4 |

### Random misclassified test queries (`distilbert`)

| text | true | predicted | conf |
|---|---|---|---|
| How do I know when my card will arrive? | card_arrival | card_delivery_estimate | 0.92 |
| Where did this fee come from? | extra_charge_on_statement | card_payment_fee_charged | 0.78 |
| What countries can I travel to and have access to their currencies? | fiat_currency_support | country_support | 0.88 |
| I would like to refill my account using SWIFT. | top_up_by_bank_transfer_charge | request_refund | 0.70 |
| Can I get a copy of the card by email? | getting_virtual_card | card_linking | 0.52 |
| The app wouldn't accept my top up. | top_up_reverted | top_up_failed | 0.87 |
| How do I freeze my card using the app? | compromised_card | lost_or_stolen_phone | 0.31 |
| Is there places where I can't withdraw money? | atm_support | declined_cash_withdrawal | 0.52 |
| I was attempting to purchase a golf club off eBay yesterday, but my credit card was declined. I tried multiple times, and again this morning. Can you check into my card please? | declined_transfer | declined_card_payment | 0.59 |
| I bought something and returned it and the money from the return isn't in my account. | Refund_not_showing_up | reverted_card_payment? | 0.69 |
| When will the transfer go through? | pending_transfer | transfer_not_received_by_recipient | 0.37 |
| How much does it cost in fees to use your card? | order_physical_card | card_payment_fee_charged | 0.94 |
| can I have a non virtual card? | order_physical_card | getting_virtual_card | 0.86 |
| How long does a UK transfer take? | balance_not_updated_after_bank_transfer | transfer_timing | 0.92 |
| What is needed to prove my identity? | verify_my_identity | unable_to_verify_identity | 0.95 |