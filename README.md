# *RAGs to Riches:* A CS426 Ablation Study of Medical RAG Pipeline Through Summarization Model Comparison
#### *By: Jack Skupien, Matthew Fecco, and William Greenwood*
*\*This notebook was written for for CS426: Data Mining and Analytics*
## Description:
This project aims to explore, compare, and contrast the performances of different summarization models by applying them to unstructured discharge papers as an ablation study for a RAG pipeline. The initial data came from unstructured ICU discharge reports in the [MIMIC-III dataset](https://physionet.org/content/mimiciii/1.4/) file NOTEEVENTS.csv with creditial access and basic training in handling human subject studies required. Each of these notes is de-identified and publicly accessible upon request. It uses PySpark to extract, filter, and parse a subset of around 1,000 of these reports, after which, it summarizes them at 90% compression rate using segmented summarization via the following abstractive models:
* **BART Model**: sequence-to-sequence text generation
    * Parameter count: 406M
* **T5 Model**: text-to-text, encoder-decoder transformer
    * Parameter count: 220M
* **Pegasus X**: Extracted Gap sentence-to-sentence model
    * Parameter count: 568M
* **Gemini 2.0 Flash**: General multimodal LLM 
    * Parameter count: undisclosed (estimated > 40B)

Once thoroughly summarized, we calculate  and compare several performance metrics across these different model sets, including:
* **Bleu scores** - using pyspark
* **Rouge scores** - using pyspark
* **BERT scores** (F1, precision, recall)
* **Pairwise cosine similarity** within RAG embedding model

Finally, we compile all this data into the following visualizations, in order to conduct a visual analysis:
* **PCA Plot embeddings** in 2D and 3D variants
* **Heatmaps** for Bleu/Rouge scores
* **Bar graphs** for BERTscore metrics
* **Word plots** to look for vocabulary diversity and word choice patterns.
* **Violin/Tukey plots** for pairwise cosine similarity distribution, mean, median, and outliers

## Installation Instructions:
All of these tasks are structured as different Python-based files that can be run locally on any computer if pulled from this repository. Please see the *Use* Section below for more detailed instructions on how to run these files.

## Use:
*Note: the following instructions are made with a Unix-based computer in mind (specifically a Macbook Pro); they would likely change based on your machine*
#### Required installs
* **Python (3.11+ recommended)**<br>
    - Can be Installed from [the Python website](https://www.python.org)
* **PySpark**
    - Can be installed by running the following command:<br> `$ pip install pyspark`
* **PyTorch & Sci-Kit Learn**
    - Can be installed by running the following commands:<br> `$ pip install torch`<br>`$ $ip install scikit-learn`
* **Pandas & NumPy**
    - Most Python installs come with these packages natively; otherwise, they can be installed by running the following commands:<br> `$ pip install pandas`<br> `$ pip install numpy`
    - Further documentation on [the Pandas website](https://pandas.pydata.org/docs/getting_started/install.html) and [the Numpy wesbite](https://numpy.org)
* **Transformers (4.51+ recommended)**
    - Can be installed from [PyPi's website](https://pypi.org/project/transformers/) or by running the following command:<br> `$ pip install transformers`
* **NLTK**
    - Documentation can be found on [NLTK's website](https://www.nltk.org/install.html)
    - Can be installed by running the following command:<br> `$ pip install nltk`
* **Scoring**
    - Further Documentation can be found on [PyPi](https://pypi.org/project/bert-score/)
    - Can be installed by running the following commands:<br> `$ pip install bert_score`<br>`$ pip install rouge-score`
* **Homebrew (recommended for Java Environment Install)**
    - Can be installed by running the following command:<br>`$ pip install brew`
* **Matplotlib (3.10+ recommended) and Seaborn**<br>
    - Can be Installed with the below commands <br>`$ python -m pip install -U matplotlib`<br>`$ pip install seaborn`
    - Further installation documentation can be found on [the matplotlib website](https://matplotlib.org/stable/install/index.html)
* **WordCloud**<br>
    - Can be Installed with the below commands <br>`$ pip install wordcloud`
    - Further installation documentation can be found on [PyPi](https://pypi.org/project/wordcloud/)
* **OpenJDK 11**
    - Can be installed with:<br>`$ brew install openjdk@11`
    - further setup may be required to symlink and set as default java version:
        1. `$ sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk`
        2. `$ echo 'export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc`
        3. `$ echo 'export JAVA_HOME="$(/usr/libexec/java_home -v 11)"' >> ~/.zshrc`
        4. `$ source ~/.zshrc`
    - you can verify that OpenJDK 11 has been set to your default version with the following command:
        `$ java -version`
    
        which should respond with something like the following:
        > openjdk version "11.0.26" 2025-01-21
            OpenJDK Runtime Environment Homebrew (build 11.0.26+0)
            OpenJDK 64-Bit Server VM Homebrew (build 11.0.26+0, mixed mode)

Once pulled with the correct libraries installed, it'll prompt you to pick a kernel (pick your python version). From there, this file can be run the same as any other Jupyter notebook. I prefer to use the Jupyter notebook extension on Visual Studio Code, but you can also use Anaconda Navigator. Either Way, each cell can be executed with its play button.

<!-- [VSCODE EXAMPLE](Screenshot%202025-02-14%20at%203.43.27 PM.png)

[ANACONDA EXAMPLE](Screenshot%202025-02-14%20at%204.09.10 PM.png) -->

<!-- ## Module Requirements
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
    pip install seaborn -->


### Summarization_and_scoring.ipynb
This notebook contains checkpoints for each step where local files can be loaded and does not require all cells to run for different execution blocks.

* Contains code for initial data extraction, cleaning, and summariztion from MIMIC-III dataset. Initial work with dataset is saved using csv files for continued interaction because NOTEEVENTS.csv is very large file.

* NOTEEVENTS.csv is not present in our repository because it is a protected file that requires credintial access and additional training. We do not believe it is ethical to keep raw dataset on public repo. 

* Summarization tasks are executed with segmented approach. Notebook contains code for both MapReduce execution or traditional exectuion. All sumarizations are saved to csv files.

* Bleu, Rouge, and BERTscores are compiled and saved to csv files for further work and visializtions

* Summaries and raw reports are embedded into RAG pipline and vectors saved to csv files.

* Wordclouds are generated for qualitative vocabulary analysis.

### cosine_similarity.ipynb
This notebook contains pairwise cosine similarity calculations and visualiztions. Comparing embedding vectors for summaries against first 512 tokens of raw report in chosen RAG pipline's embedding model.

### visualizations_new.ipynb*
This notebook contains embedding PCA visualizations and cleaning functions for better csv format.

**\*NOTE: For more immersive individual 3D visualizations, the separate model visualization scripts will produce visualizations that you can pan, zoom, rotate, etc.**

### cosine_distances.ipynb
This notebook contains cosine distance calculations for embedding cluster sparseness and visualizations.

### score_visuals.ipynb
This notebook contains visualizations for Bleu, Rouge, and BERTscore metrics.


## Contact:
Please feel free to email the authors of this Repository:
* Jack Skupien
jskupien@vols.utk.edu
* Matt Fecco
mfecco@vols.utk.edu
* William Greenwood
wgreenwo@vols.utk.edu