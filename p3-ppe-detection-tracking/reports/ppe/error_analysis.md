Variant `pretrained_n` seed 0, conf >= 0.25, IoU >= 0.5

### Recall and false positives per class

| class | GT | recall | FP background | FP localization | FP class | FP duplicate |
|---|---|---|---|---|---|---|
| helmet | 294 | 0.799 | 5 | 3 | 14 | 0 |
| gloves | 381 | 0.753 | 17 | 19 | 7 | 1 |
| vest | 340 | 0.832 | 0 | 2 | 19 | 0 |
| boots | 481 | 0.703 | 11 | 37 | 37 | 1 |
| goggles | 139 | 0.942 | 3 | 4 | 2 | 1 |
| Person | 481 | 0.898 | 17 | 11 | 4 | 0 |
| no_helmet | 179 | 0.547 | 3 | 8 | 1 | 0 |
| no_goggle | 82 | 0.220 | 0 | 12 | 0 | 2 |
| no_gloves | 203 | 0.581 | 3 | 8 | 1 | 2 |
| no_boots | 178 | 0.646 | 1 | 1 | 8 | 0 |
| forklift | 73 | 0.781 | 7 | 3 | 0 | 0 |
| no_vest | 130 | 0.615 | 0 | 3 | 3 | 0 |

### Recall by box size (COCO buckets, original pixels)

| size | GT | recall |
|---|---|---|
| small | 511 | 0.174 |
| medium | 804 | 0.672 |
| large | 1646 | 0.950 |

### Most frequent class confusions (FP labelled with the wrong class)

| true | predicted | count |
|---|---|---|
| no_boots | boots | 36 |
| no_helmet | helmet | 14 |
| no_vest | vest | 13 |
| boots | no_boots | 8 |
| no_gloves | gloves | 7 |
| Person | vest | 5 |
| vest | no_vest | 3 |
| no_vest | Person | 2 |
| forklift | Person | 1 |
| helmet | vest | 1 |

### Compliance rule: GT vs predictions (481 GT persons, agreement 0.696)

| GT status | predicted status | count |
|---|---|---|
| compliant | compliant | 188 |
| compliant | missed | 7 |
| compliant | non_compliant | 6 |
| compliant | unknown | 29 |
| non_compliant | compliant | 14 |
| non_compliant | missed | 28 |
| non_compliant | non_compliant | 138 |
| non_compliant | unknown | 47 |
| unknown | missed | 14 |
| unknown | non_compliant | 1 |
| unknown | unknown | 9 |
