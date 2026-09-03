# Historical conference split vs proposed journal_v1

This is a protocol and composition comparison only. No model-performance comparison was performed. The historical results remain preserved as originally evaluated.

| Partition | Conference images (%) | journal_v1 images (%) | Conference lesions | journal_v1 lesions |
|---|---:|---:|---:|---:|
| train | 8013 (80.0100%) | 8011 (79.9900%) | 6274 | 5975 |
| validation | 1001 (9.9950%) | 1002 (10.0050%) | 967 | 747 |
| test | 1001 (9.9950%) | 1002 (10.0050%) | 976 | 748 |

## Lesion overlap

| Boundary | Conference shared lesion IDs | journal_v1 shared lesion IDs |
|---|---:|---:|
| train / validation | 357 | 0 |
| train / test | 362 | 0 |
| validation / test | 50 | 0 |
| all three | 22 | 0 |

## Class composition

| Partition | dx | Conference count (%) | journal_v1 count (%) | Difference (percentage points) |
|---|---|---:|---:|---:|
| train | akiec | 261 (3.2572%) | 262 (3.2705%) | +0.0133 |
| train | bcc | 412 (5.1416%) | 411 (5.1304%) | -0.0112 |
| train | bkl | 879 (10.9697%) | 879 (10.9724%) | +0.0027 |
| train | df | 91 (1.1357%) | 92 (1.1484%) | +0.0128 |
| train | mel | 891 (11.1194%) | 890 (11.1097%) | -0.0097 |
| train | nv | 5365 (66.9537%) | 5363 (66.9455%) | -0.0083 |
| train | vasc | 114 (1.4227%) | 114 (1.4230%) | +0.0004 |
| validation | akiec | 33 (3.2967%) | 32 (3.1936%) | -0.1031 |
| validation | bcc | 51 (5.0949%) | 51 (5.0898%) | -0.0051 |
| validation | bkl | 110 (10.9890%) | 110 (10.9780%) | -0.0110 |
| validation | df | 12 (1.1988%) | 12 (1.1976%) | -0.0012 |
| validation | mel | 111 (11.0889%) | 112 (11.1776%) | +0.0887 |
| validation | nv | 670 (66.9331%) | 671 (66.9661%) | +0.0330 |
| validation | vasc | 14 (1.3986%) | 14 (1.3972%) | -0.0014 |
| test | akiec | 33 (3.2967%) | 33 (3.2934%) | -0.0033 |
| test | bcc | 51 (5.0949%) | 52 (5.1896%) | +0.0947 |
| test | bkl | 110 (10.9890%) | 110 (10.9780%) | -0.0110 |
| test | df | 12 (1.1988%) | 11 (1.0978%) | -0.1010 |
| test | mel | 111 (11.0889%) | 111 (11.0778%) | -0.0111 |
| test | nv | 670 (66.9331%) | 671 (66.9661%) | +0.0330 |
| test | vasc | 14 (1.3986%) | 14 (1.3972%) | -0.0014 |

The proposed journal_v1 split retains approximately 80/10/10 image proportions and closely tracks class proportions while making lesion IDs mutually exclusive. It is intended for future research evaluation and does not overwrite or relabel the historical conference protocol.
