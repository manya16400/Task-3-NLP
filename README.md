# Task-4.-NLP
The entire code has been developed using Python programming language, utilizing its powerful text processing and machine learning modules.

### About
This repository illustrates the analysing, text preprocessing/cleaning of flipkart product data and building a classifier to classify the products into their respective categories.

### Data
After applying preprocessing steps to the dataset we are left with ~19k samples. Splitting into train, validation and test (70%, 20% and 10%).

### Choosing the category from product_category_tree
Applied two techniques to figure out the primary category of each sample - <br/>
1. Chose the main category of product_category_tree as the label.
2. Found out the frequency of all the unique categories in the train+valid dataset and chose the ones which have frequency greater than equal to 100. 
  Sorted the list on the basis of increasing order of frequency.
  Traversed through each sample's product_category_tree and chose the category which ocuurs in the sorted list.
  
### Classifier
#### Approach:
 1. The text features are cleaned by removing bad symbols and stopwords using nltk.
 2. Considered two types of features as input- <br/>
    a) Decription
    b) Product_name, description, brand, product_specifications
 3. Applied sklearn feature_extraction libraries CountVectorizer() and TfidfTransformer() (term frequency-inverse document frequency formula) and word2vec to our features to convert text into embeddings.
 4. The following algorithms are applied on the dataset- <br/>
     a) Naive-Bayes Classifier
     b) Random Forest
     c) Linear SVM
     d) MLP Classifier
     e) Logistic REgression
     f) Bert Classifier
     
## Results  
### 1. Main category as label
   
  FEATURE | MODEL | VECTORIZER | ACCURACY
  --------|-------|------------|---------
  __Description__|NB Classifier|TF-IDF|86.61
  __Description__|Linear SVM|TF-IDF|94.99
  __Description__|__Logistic Regression__|__TF-IDF__|__98.21__
  __Description__|MLP Classifier|TF-IDF|97.32
  __Description__|Logistic Regression|word2vec|93.14
  __Description__|Random Forest|word2vec|89.78
  __Description__|MLP Classifier|word2vec|93.17
  __Product_name, description, brand, product_specifications__|NB Classifier|TF-IDF|87.27
  __Product_name, description, brand, product_specifications__|Linear SVM|TF-IDF|95.27
  __Product_name, description, brand, product_specifications__|Logistic Regression|TF-IDF|98.15
  __Product_name, description, brand, product_specifications__|MLP Classifier|TF-IDF|97.52
  __Product_name, description, brand, product_specifications__|Logistic Regression|word2vec|94.55
  __Product_name, description, brand, product_specifications__|Random Forest|word2vec|89.03
  __Product_name, description, brand, product_specifications__|MLP Classifier|word2vec|93.58
  
   FEATURE | BERT
   ---------|----
   __Description__|96.98
   __Product_name, description, brand, product_specifications__ |97.09  
   
### 2. Category based on most common categories in the dataset as label
   
  FEATURE | MODEL | VECTORIZER | ACCURACY
  --------|-------|------------|---------
  __Description__|NB Classifier|TF-IDF|67.76
  __Description__|Linear SVM|TF-IDF|87.99
  __Description__|Logistic Regression|TF-IDF|94.70
  __Description__|MLP Classifier|TF-IDF|89.86
  __Description__|Linear SVM|word2vec|77.48
  __Description__|Logistic Regression|word2vec|90.29
  __Description__|MLP Classifier|word2vec|83.16
  __Product_name, description, brand, product_specifications__|NB Classifier|TF-IDF|71.73
  __Product_name, description, brand, product_specifications__|Linear SVM|TF-IDF|88.31
  __Product_name, description, brand, product_specifications__|Logistic Regression|TF-IDF|95.33
  __Product_name, description, brand, product_specifications__|MLP Classifier|TF-IDF|91.59
  __Product_name, description, brand, product_specifications__|Linear SVM|word2vec|77.11
  __Product_name, description, brand, product_specifications__|Logistic Regression|word2vec|91.33
  __Product_name, description, brand, product_specifications__|MLP Classifier|word2vec|83.47
  
   FEATURE | BERT
   ---------|----
   __Description__|81.69
   __Product_name, description, brand, product_specifications__ |97.09  
   
__Inferences__-
1. After performing the experiments, it is observed that Product_name, description, brand and product_specifications together as feature performs better than just description as a feature which implies that other information along with description is necessary as well for a better classification model.
2. When the main category is considered as label, __Logistic Regression__ with __description__ as feature with __TF-IDF__ as vectorizer out performs all the other models with an accuracy of __98.21%__.
3. When category based on most common categories in the dataset as label, __Logistic Regression__ with __Product_name, description, brand, product_specifications__ as feature and __word2vec__ as vectorizer out performs all the other models with an accuracy of __95.33%__.
   
