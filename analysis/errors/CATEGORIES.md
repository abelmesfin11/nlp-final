# Error categories for manual labeling

Put ONE key in the `category` column (add a second with `;` only if truly needed).

- `negation` - sentiment flipped by not / never / no / n't (e.g. 'not bad')
- `contrast` - mixed sentiment with but / although / however; model follows the wrong clause
- `sarcasm` - irony or sarcasm; literal words disagree with intended meaning
- `comparison` - compares to other films ('no Citizen Kane, but ...')
- `world_knowledge` - needs knowledge of actors, genres, references
- `rare_words` - unusual vocabulary, idioms, slang the model misreads
- `neutral_ambiguous` - no clear sentiment; a human could go either way
- `label_noise` - the GOLD label itself looks wrong
- `plot_vs_opinion` - (IMDb) describes plot events that sound negative/positive but the review is the opposite
- `truncation` - (IMDb) the deciding sentence is past the 512-token cutoff
- `other` - anything else - describe in notes
