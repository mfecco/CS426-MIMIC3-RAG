# *RAGs to Riches:* A CS426 Ablation Study of Medical RAG Pipeline Through Summarization Model Comparison

We want to explore different summarization models for unstructured text notes as an ablation study for a RAG pipeline.

* Initial data from unstructured discharge reports in MIMIC-III dataset file NOTEEVENTS.csv with creditial access and basic training in handling human subject studies required.
* Use pyspark.sql to extract, filter, and clean notes.
* Summarize notes at 90% compression rate using segmented summarization with abstractive models.
    * 4 different sets of the same ~1000 discharge reports: 
        * BART model: sequence-to-sequence text generation
            * Parameter count: 406M
        * T5 model: text-to-text, encoder-decoder transformer
            * Parameter count: 220M
        * Pegasus x: Extracted Gap sentence-to-sentence model
            * Parameter count: 568M
        * Gemini 2.0 flash: General multimodal LLM 
            * Parameter count: undisclosed (estimated > 40B)
* Calculate the Bleu and Bart_score scores of each summarization and compare values accross different model sets.
    * Bleu scores - using pyspark
    * Rouge score - using pyspark
    * BERTscores F1, precision, recall
* Visual Analysis
    * Plot embeddings using PCA
    * Heatmaps for Bleu/Rouge scores
    * Bar graph for BERTscore metrics
    * Word plots to look for vocabulary diversity and word choice patterns.



## Module Requirements
    pip install pyspark
    pip install torch
    pip install transformers
    pip install pandas
    pip install nltk
    pip install rouge-score
    pip install bert_score
    pip install wordcloud
    pip install numpy
    pip install scikit-learn
    pip install matplotlib
    pip install seaborn


## Summarization_and_scoring.ipynb
This notebook contains checkpoints for each step where local files can be loaded and does not require all cells to run for different execution blocks.

* Contains code for initial data extraction, cleaning, and summariztion from MIMIC-III dataset. Initial work with dataset is saved using csv files for continued interaction because NOTEEVENTS.csv is very large file.

* NOTEEVENTS.csv is not present in our repository because it is a protected file that requires credintial access and additional training. We do not believe it is ethical to keep raw dataset on public repo. 

* Summarization tasks are executed with segmented approach. Notebook contains code for both MapReduce execution or traditional exectuion. All sumarizations are saved to csv files.

* Bleu, Rouge, and BERTscores are compiled and saved to csv files for further work and visializtions

* Summaries and raw reports are embedded into RAG pipline and vectors saved to csv files.

* Wordclouds are generated for qualitative vocabulary analysis.

## cosine_similarity.ipynb
This notebook contains pairwise cosine similarity calculations and visualiztions. Comparing embedding vectors for summaries against first 512 tokens of raw report in chosen RAG pipline's embedding model.

## visualizations_new.ipynb
This notebook contains embedding PCA visualizations and cleaning functions for better csv format.

## cosine_distances.ipynb
This notebook contains cosine distance calculations and visualizations

## score_visuals.ipynb
This notebook contains visualizations for Bleu, Rouge, and BERTscore metrics


## For Questions
contact: mfecco@vols.utk.edu