### Abstract:

Telemarketing is one of the convenient and effective methods of selling products and services to the customer. If not targeted to the right customer, these telemarketing calls might be perceived as irritating which might instead decrease the company’s value. Using a Portuguese bank telemarketing dataset, we implemented various machine learning algorithms to predict the right customer. We utilized an over-sampling method called SMOTE to mitigate the class imbalance problem. The Light Gradient Boosting Machine (LGBM) model on the plain dataset scored highest AUC of 0.80 then the over-sampled dataset, implicating that SMOTE might not add any benefits to complex ensemble tree methods. For model explainability, we implemented the global as well as local explainer to streamline the decision-making process. 

### Dataset:

The dataset is open-source and can be downloaded from the [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The original paper that open-sourced data can be found  at:

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62,* 22-31.

The dataset is imbalced consistingof 4,639 'yes' and 36,549 'no'. 

### Methodology:

The methodology section is sub-divided into following section:

1. Pre-Processing 
    - Removed the varibale like 'pdays', 'default' due to the lack of variability.
    - Removed column 'default' as suggested by author. 
    - Encoded the nominal and ordinal value to one-hot encoding and ordinal encoding respectively.
    - The conitnuous data were standardized using Z-score.

2. Train-Test Split & Cross Validation
   - 90% Train, 10% Test
   - Train data is further divided into 10 Fold Cross-Validation. 
   - Conducted statrified split to preserve the same ratio of target class.

3. SMOTE
   - SMOTE is utilized to tackle class imbalance problem. 
   - Generates synthetic data to balance the class on training set. 

4. Models Used
   - Random Forest (RF), Logistic Regression (LR), Extra Tree Classifier (ET), Xtream Gradient Boosting Machines (XGBM), Light Gradient Boosting Machines (LGBM), CatBoost (CB).

5. Evaluation Metrics
   - Precision (PR), Recall (RC), ROC-AUC (Area Under Curve), F1-ratio, Accuracy (ACC), Matthews Correalation Coefficient (MCC)
   - AUC score will be used to select the best model.

### Results

- The experiment are divided into two parts. One with the plain data (no SMOTE) and other with SMOTED data. 

|            |     Plain   Dataset    |               |               |               |                |               |     SMOTE   Dataset    |               |               |               |               |               |
|------------|:----------------------:|:-------------:|:-------------:|:-------------:|:--------------:|:-------------:|:----------------------:|---------------|---------------|---------------|---------------|---------------|
|            |            LR          |       RF      |       ET      |      XGBM     |       LGBM     |       CB      |            LR          |       RF      |       ET      |      XGBM     |      LGBM     |       CB      |
|     ACC    |     0.9001             |     0.8918    |     0.8801    |     0.8984    |     0.9006*    |     0.8992    |     0.866              |     0.8837    |     0.873     |     0.898     |     0.8964    |     0.8973    |
|     AUC    |     0.7912             |     0.7726    |     0.7455    |     0.7875    |     0.8005*    |     0.799     |     0.7796             |     0.7732    |     0.7455    |     0.7845    |     0.7945    |     0.7904    |
|      RC    |     0.2262             |     0.2945    |     0.3063    |     0.2876    |     0.2677     |     0.2773    |     0.5423*            |     0.3689    |     0.3667    |     0.3142    |     0.3205    |     0.3032    |
|     PRC    |     0.6646*            |     0.5349    |     0.4519    |     0.6043    |     0.6398     |     0.6179    |     0.4257             |     0.4784    |     0.426     |     0.5892    |     0.5719    |     0.5859    |
|      F1    |     0.3372             |     0.3796    |     0.365     |     0.3891    |     0.377      |     0.3824    |     0.4768*            |     0.4163    |     0.3939    |     0.4095    |     0.4104    |     0.3992    |
|     MCC    |     0.3484             |     0.3434    |     0.3085    |     0.3697    |     0.3706     |     0.3685    |     0.4053*            |     0.3567    |     0.3248    |     0.3809    |     0.3769    |     0.3722    |



*Note: * denotes that the value is highest among two different experimentation across various evaluation metrics.* 

- The LGBM Model without SMOTING performed best among others. 
- This implicates that for complex ensemble models SMOTE might not add any benefits (Elor & Avrbucg-Elor, 2022). 

### Threshold Optimization

- In general, the threshold is set to be 0.5, and so was ours, which means that the predicted probability of an instance that is greater than 0.5 is flagged as positive and less than 0.5 as negative.
- We ran several experiments on the LGBM model to find the optimal cut-off value. 

![Threshold Optimization](graphs\threshold.png)

- From figure, we found that 0.19 are our optimal model. 
- We trained new model with that threshold and evaluated on the test data.

|             |       ACC     |       AUC     |       RC      |       PRC     |       F1      |       MCC     |
|-------------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|     LGBM    |     0.8715    |     0.8055    |     0.5680    |     0.4443    |     0.4986    |     0.4304    |

- The Precison and Recall has incresed along with F1 and MCC. 
  
### Explainability

- Feature Importance 
- Local Interpretable Model-Agnostic Explanations (LIME)

### Deployment

- Deployed the app that gives marketers ability to input customer's data in order to get the probability of either buying or rejecting the sales call. 
- The app also shows the features, with the help of LIME, that effects the consumers decision. 

![Web App Screenshot](graphs\web_app.png)

- The deployed app can be accessed via [link]()

### Conslusion
Even in the era of social media marketing, telemarketing still stands out with its own advantages. Our results show that the plain model outperformed the SMOTED model. The LGBM model achieved an AUC score of 0.8055 on the test set. Our comparison validates previous findings that SMOTE might not be valuable when used in more complex ensemble tree algorithms (Elor and Averbuch-Elor 2022). However, it might add some value to a simple tree or linear model. Explainability remains at the center of any predictive modeling when comes to business decision-making.
We implemented a feature importance plot for global explainability. However, the feature importance plot doesn’t give us the ability to further investigate a specific customer. Therefore, LIME is implemented to explain every single observation. To our knowledge, there hasn’t been any research done in a telemarketing domain where LIME was implemented to explore each instance. A robust predictive model with
interpretability enables identifying the right customer and avoiding annoying calls to uninterested customers. The interpretable methods also help to understand the consumers’ intention so that the business personal can make data-driven unbiased decisions. Such insight will help company to attract the right customer eventually contributing to the company’s growth.

### Libraries Used
Plese refer to the [`requirements.txt`]() file. 