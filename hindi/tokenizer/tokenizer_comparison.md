# Hindi Unigram Tokenizer Comparison Report

All tokenizers were trained **from scratch** using SentencePiece Unigram on the entire Hindi training dataset (`hindi/data/splits/train.parquet`).

## 1. Summary Comparison

| Vocabulary | Requested Vocab | Actual Vocab | Chars / Token | UNK Rate (%) | Model File Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **unigram_6k** | 6,000 | 6,000 | 3.4769 | 0.00% | 155.3 KB |
| **unigram_8k** | 8,000 | 8,000 | 3.8305 | 0.00% | 214.5 KB |
| **unigram_10k** | 10,000 | 10,000 | 4.0357 | 0.00% | 273.9 KB |
| **unigram_15k** | 15,000 | 15,000 | 4.4314 | 0.00% | 429.4 KB |

## 2. Sample Tokenizations Across Vocabularies

**Test Sentence:** `कृत्रिम बुद्धिमत्ता और मशीन लर्निंग आज महत्वपूर्ण तकनीकें हैं।`

| Vocabulary | Tokens Count | Segmented Pieces |
| :--- | :--- | :--- |
| **unigram_6k** | 18 | `▁` `कृत` `्रि` `म` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।` |
| **unigram_8k** | 18 | `▁` `कृत` `्रि` `म` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।` |
| **unigram_10k** | 15 | `▁कृत्रिम` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।` |
| **unigram_15k** | 15 | `▁कृत्रिम` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।` |
