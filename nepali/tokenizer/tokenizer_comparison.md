# Nepali Unigram Tokenizer Comparison Report

All tokenizers were trained **from scratch** using SentencePiece Unigram on the entire Nepali training dataset (`nepali/data/splits/train.parquet`).

## 1. Summary Comparison

| Vocabulary | Requested Vocab | Actual Vocab | Chars / Token | UNK Rate (%) | Model File Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **unigram_6k** | 6,000 | 6,000 | 4.6545 | 0.00% | 159.9 KB |
| **unigram_8k** | 8,000 | 8,000 | 5.1200 | 0.00% | 220.1 KB |
| **unigram_10k** | 10,000 | 10,000 | 5.1200 | 0.00% | 280.7 KB |
| **unigram_15k** | 15,000 | 15,000 | 5.3333 | 0.00% | 438.5 KB |

## 2. Sample Tokenizations Across Vocabularies

**Test Sentence:** `कृत्रिम बौद्धिकता र प्राकृतिक भाषा प्रशोधन आजका महत्त्वपूर्ण प्रविधिहरू हुन्।`

| Vocabulary | Tokens Count | Segmented Pieces |
| :--- | :--- | :--- |
| **unigram_6k** | 15 | `▁कृत्रिम` `▁बौद्ध` `िक` `ता` `▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।` |
| **unigram_8k** | 14 | `▁कृत्रिम` `▁बौद्धिक` `ता` `▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।` |
| **unigram_10k** | 14 | `▁कृत्रिम` `▁बौद्धिक` `ता` `▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।` |
| **unigram_15k** | 13 | `▁कृत्रिम` `▁बौद्धिक` `ता▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।` |
