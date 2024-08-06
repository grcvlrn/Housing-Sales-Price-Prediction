# Housing-Sales-Price-Prediction

## I. Background
### A. Objectives:
This program is designed to predict the price of houses based on data that has been collected. Accurate house price prediction is crucial for various stakeholders in the real estate market, including buyers, sellers, and investors. By leveraging historical data, we can identify patterns and trends that influence house prices.

Using this data, the program will employ statistical and machine learning techniques to build a model that can predict house prices based on the given features. The goal is to provide accurate and reliable predictions to aid in making informed decisions in the real estate market.

### B. Data Overview:
id: Unique identifier for each house (integer)
date: Date when the house was sold (object)
price: Sale price of the house (float)
bedrooms: Number of bedrooms (integer)
bathrooms: Number of bathrooms (float)
sqft_living: Square footage of the living area (integer)
sqft_lot: Square footage of the lot (integer)
floors: Number of floors (float)
waterfront: Whether the house has a waterfront view (integer)
view: Quality of the view from the house (integer)
condition: Condition of the house (integer)
grade: Overall grade given to the housing unit, based on King County grading system (integer)
sqft_above: Square footage of the house apart from the basement (integer)
sqft_basement: Square footage of the basement (integer)
yr_built: Year the house was built (integer)
yr_renovated: Year the house was renovated (integer)
zipcode: Zip code of the house (integer)
lat: Latitude coordinate of the house (float)
long: Longitude coordinate of the house (float)
sqft_living15: Living room area (integer)
sqft_lot15: Lot area (integer)

## II. Workflow:
**Data Preprocessing:**
Handle missing values, remove duplicates, and correct any data entry errors
Scale numerical features to ensure they have a similar range
Convert categorical features into numerical formats using techniques like one-hot encoding
Create new features from existing data, such as price per square foot or proximity to amenities

**Exploratory Data Analysis (EDA):**
Create visualizations to understand the distribution of the data and relationships between features
Use statistical methods to identify significant features and correlations

**Model Selection:**
Split the dataset into training and testing sets
Choose algorithms suitable for regression tasks 
Use validation techniques to evaluate model performance

**Model Training:**
Use techniques like Grid Search or Random Search to find the best hyperparameters for your model
Train multiple models and evaluate their performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared

**Model Evaluation:**
Assess the models using the testing dataset and choose the one with the best performance
Analyze the residuals to check for patterns that the model did not capture
Ensure the model generalizes well and is not overfitting or underfitting

**Model Deployment:**
Develop an API to serve the model using frameworks like Flask or FastAPI
Integrate the model into the existing systems or applications

## III. Conclusion
First of all we will import all of the libraries that we want to use. Then we will load the data. The data have 21 columns with 21613 entries. The data have integer and float data types which are considered numerical. There is no missing value present in the data which means that the missing value handling can be skipped. Later on we drop the id and date column because it is unnecessary in predicting the price.

Then, we explored the data by analyzing it with graph and charts. We see that 45% of the sales are 3 bedroom houses. This explains that most people prefer 3 bedroom houses. The data also shows that 38% of the sales are 2 story houses. Most of the sales seem to be houses with 1-2 floors. Mainly the houses on the sales are 2.5 bathrooms with 25%. Most people here prefer 1 to 2.5 bathrooms. The sales also show that 90% of the houses don't have a good view. Which means that the buyer doesn't prioritize view, but rather functionality. The most on demand houses are the ones with about 2000 sqft building. The data also shows that people here prefer recently built houses, over the 2000s.

Next, the data are split into features that determine the prices and the price which is the target itself. The data outliers need to be handled. Therefore, we will differentiate the numerical columns with the categorical columns. It turns out that all columns are numeric columns. We will check their skewness and handle the outliers according to it. The normally distributed columns are handled with gaussian capping method and the skewed columns are handled with iqr capping method. We see from the graph, that there isn't a big difference in before and after handling the outliers. Since the data are all numerical, we only need to scale all the columns.

Next part is to test different models, the models are fitted to the features and target. First, we checked the linear regression model. The prediction shows that the performance is about 69% for both train and test data. They seem not too far off each other, which indicates stability and reliability in the model. Second, is SVM. The models seem to not be fitting well from the negative score that we see. This might be because the train data doesn't capture meaningful relationships between features and target data and so the variability in predicting is not explained well. Third, the decision tree regressor model. The performance drops slightly from the train set to the test set. It indicates the overfitting of the model. The train data from this model might not cover the unseen data in the test set. To improve this model, the data in the train set needs to be more variative. Fourth, ridge regression. The score shows that the model is consistent throughout the train and test set. To improve, this model needs varied examples for a better generalization. Last model is a random forest. This model proves to be the best one with a score performance of 79% in the train set and 77% in the test set. The data generalizes well to the unseen data in the test set. The model is proven reliable and stable in predicting results, despite the slight overfitting. The score shows us that we could trust this model in predicting the price.

We also checked the best parameters using pipeline and the grid search hyperparameter tuning. The result shows that random forest with the hyperparameter of 'regressor__max_depth': None, 'regressor__min_samples_leaf': 1, 'regressor__min_samples_split': 2, 'regressor__n_estimators': 200 works the best. The cross validation score shows negative value, which means a ood performance. The r2 score shows an 88% indicating a good fit of the data with the model. The MSE score shows that the average squared error is approximately 15.8 billion. With the type of price we are predicting, it is not too much of an error. Each fold of cross-validation shows negative value, but it iis a good indicator that the model did well. The mean cross-validation score shows -18306954260.98501 is an overall measure of model performance across different folds of the data. This confirms the model effectiveness in the prediction.


## IV. Recommendations:
##### Technique
With this experiment, I can say that the best model to use in predicting the price of house sales is random forest regressor with 'regressor__max_depth': None, 'regressor__min_samples_leaf': 1, 'regressor__min_samples_split': 2, 'regressor__n_estimators': 200 hyperparameters. The model performs 88% well.

This model is great for this data because it provides higher accuracy compared to the decision tree, which only has 1 tree. It can capture nonlinear relationships between features and target variables. Due to combining multiple trees, the model generalizes well and is less prone to overfitting. It also ranks the importance of features, which helps in feature selection. The model can handle missing value in the dataset.

Though, the model training computation is quite time consuming especially in large dataset and large number of trees. It is also harder to interpret compared to a single tree model. The model also requires more memory usage due to the number of trees.

By carefully tuning parameters, optimizing features, and considering ensemble methods, you can leverage the strengths of random forests while reducing some of its downside to achieve better performance in regression tasks.

##### Business
By creating this model to predict housing price, we can implement strategies such as:

Targeted Marketing and Sales: The model will identify areas where housing prices are predicted to rise and we could hold marketing campaigns with personalized offers to potential buyers in those areas.

Investment and Development: We cold acquire land or properties in regions where the model expect the housing price rise to develop houses or resale at a higher price. Also, adjust insurance premiums and mortgage lending rates based on market trends with downturns or areas with stagnant growth for investment strategy.

Pricing Strategies: To adjust the price of houses appropriately according to market conditions to stay competitive and maximize profit margins by using this model.

Inventory Management: Managing property inventories by optimizing the timing of property sales where market peaks to align supply with demand based on trends.

Customer Insights: With this model, we could understand customers' preferences to offer realistic prices.

Partnership and Alliances: We could form strategic partnerships with builders and developers, collaborate with real estate agents, and mortgage lenders to create projects in high-growth areas by offering bundled services based on market predictions.

Innovation and Technology: Appealing to tech-savvy and environmentally conscious buyers with smart home technology and sustainable and energy-efficient building practices in high-demand and increasing housing price areas.

## IV. Attachment
[Powerpoint](https://docs.google.com/presentation/d/18wHodnB8ODaqO8ixMLs7lDFKTi392xJvicXd2BtRCOI/edit?usp=sharing)
[Model.pkl](https://drive.google.com/file/d/1mKrC1KieeDg0zJBV2_c3oa_rg6bCIJYa/view?usp=sharing)
[Huggingface Deployment](https://huggingface.co/spaces/grcvlrn/Price_Prediction_Deployment/tree/main)
