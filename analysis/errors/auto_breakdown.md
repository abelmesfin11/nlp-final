# Automatic error breakdown
Error rates (%) within slices flagged by simple regexes. Use these to choose what to
look at by hand - they are hints, not conclusions.

### SST-2, 0% noise

| Slice | n | full err% | lora err% | dora err% |
|---|---|---|---|---|
| all | 872 | 5.5 | 5.2 | 5.6 |
| has negation | 190 | 5.8 | 5.3 | 5.8 |
| has contrast word | 161 | 6.2 | 6.2 | 6.8 |
| short (<=10 words) | 152 | 5.9 | 5.9 | 5.9 |
| long (>25 words) | 225 | 3.6 | 4.0 | 4.0 |

### IMDb, 0% noise

| Slice | n | full err% | lora err% | dora err% |
|---|---|---|---|---|
| all | 2000 | 11.4 | 10.4 | 10.8 |
| has negation | 1769 | 11.6 | 10.7 | 11.0 |
| has contrast word | 1639 | 12.4 | 11.2 | 11.5 |
| short (<=150 words) | 813 | 7.5 | 6.8 | 7.4 |
| long (>400 words, likely truncated) | 226 | 25.2 | 26.5 | 26.1 |

### SST-2, 30% noise

| Slice | n | full err% | lora err% | dora err% |
|---|---|---|---|---|
| all | 872 | 15.7 | 11.6 | 13.4 |
| has negation | 190 | 18.9 | 16.3 | 16.3 |
| has contrast word | 161 | 16.8 | 14.9 | 18.0 |
| short (<=10 words) | 152 | 13.8 | 7.9 | 13.2 |
| long (>25 words) | 225 | 17.8 | 16.0 | 16.0 |

### IMDb, 30% noise

| Slice | n | full err% | lora err% | dora err% |
|---|---|---|---|---|
| all | 2000 | 17.9 | 16.2 | 15.7 |
| has negation | 1769 | 18.2 | 16.5 | 15.8 |
| has contrast word | 1639 | 18.9 | 17.3 | 16.6 |
| short (<=150 words) | 813 | 14.3 | 12.9 | 13.7 |
| long (>400 words, likely truncated) | 226 | 38.5 | 26.5 | 27.0 |

### Memorization, 30% noise

Share of corrupted training examples fit to the WRONG label: full=84.5%, lora=46.0%, dora=54.9%

Overlap (how many methods memorized each example): 0 methods: 843, 1 methods: 1583, 2 methods: 1178, 3 methods: 2396

Mean length of memorized vs not (words), per method:

- full: memorized 9.1, not memorized 11.5
- lora: memorized 9.1, not memorized 9.8
- dora: memorized 9.0, not memorized 10.0
