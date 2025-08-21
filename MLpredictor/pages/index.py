import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Home - My App",
    page_icon="🏠",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background-color: #ff69b4;
            font-family: Arial, sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #2C3E50;
            padding-top: 30px;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
            margin-bottom: 50px;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
            padding: 8px;
            border-radius: 5px;
        }
        .independent-label {
            background-color: #d4f1f9;
            color: #0b7285;
        }
        .dependent-label {
            background-color: #ffe3e3;
            color: #c92a2a;
        }
        .split-label {
            background-color: #e8f5e9;
            color: #1b5e20;
        }
        .stButton>button {
            background-color: #3498DB;
            color: white;
            border-radius: 8px;
            height: 3em;
            font-size: 16px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #2980B9;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- MAIN TITLE ----------
st.markdown("<div class='main-title'>Welcome to My App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your one-stop solution for predictions, data analysis, and more</div>", unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(" Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # ---------- VARIABLE SELECTION ----------
    st.markdown("<div class='section-title independent-label'>Select Independent Variables (Features)</div>",
                unsafe_allow_html=True)
    independent_vars = st.multiselect(
        "Independent Variables:",
        options=df.columns.tolist()
    )

    st.markdown("<div class='section-title dependent-label'>Select Dependent Variable (Target)</div>",
                unsafe_allow_html=True)
    dependent_vars = st.selectbox(
        "Dependent Variable:",
        options=df.columns.tolist()
    )

    # --- Validation: Ensure both are selected ---
    if not independent_vars or not dependent_vars:
        st.warning("⚠️ Please select both Independent and Dependent variables to proceed.")
    else:
        st.success("✅ Independent and Dependent variables selected successfully!")

        # ---------- DEFINE X AND y ----------
        X = df[independent_vars].copy()
        y = df[dependent_vars].copy()

        # ---------- OPTIONAL CATEGORICAL ENCODING ----------
        st.markdown(
            "<div class='section-title' style='background-color:#fff3cd; color:#856404;'>Optional: Encode Categorical Variables</div>",
            unsafe_allow_html=True)
        encode_categorical = st.checkbox(
            "Encode categorical columns (Label + One-Hot Encoding) and avoid dummy variable trap"
        )

        if encode_categorical:
            categorical_cols = st.multiselect(
                "Select categorical columns to encode",
                options=X.columns.tolist()
            )

            if categorical_cols:
                ct = ColumnTransformer(
                    transformers=[
                        ("encoder", OneHotEncoder(drop='first'), categorical_cols)
                    ],
                    remainder='passthrough'
                )
                X_encoded = ct.fit_transform(X)
                X = pd.DataFrame(X_encoded)
                st.success("Categorical encoding applied successfully!")
                st.dataframe(X)
            else:
                st.warning("No categorical columns selected for encoding.")
        else:
            st.info("Categorical encoding skipped.")

        # ---------- OPTIONAL FEATURE SCALING ----------
        scale_features = st.checkbox("Apply Feature Scaling", value=False)
        if scale_features and X is not None:
            sc_x = StandardScaler()
            X = sc_x.fit_transform(X)

            if y is not None and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_integer_dtype(y):
                sc_y = StandardScaler()
                y = sc_y.fit_transform(y.to_numpy().reshape(-1, 1))


        # ---------- OPTIONAL DATA SPLITTING ----------
        if st.checkbox(" Split dataset into Training and Testing sets"):
            st.markdown("<div class='section-title split-label'> Split the Dataset</div>", unsafe_allow_html=True)
            test_size = st.slider("Select Test Size (%)", min_value=10, max_value=50, value=20, step=5)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=0
            )
            # Save original y before scaling (for classification task later)
            y_train_original, y_test_original = y_train.copy(), y_test.copy()

            st.success(f" Dataset split complete! Training set: {len(X_train)} rows, Test set: {len(X_test)} rows.")
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
            st.warning("⚠ No train-test split applied. Proceeding with full dataset.")
    # ---------- MACHINE LEARNING OPTIONS ----------
    st.markdown("### ⚙ Select Machine Learning Task")
    task = st.selectbox(
        "Choose a Task",
        ["Regression", "Classification", "Clustering"]
    )

    if task == "Regression":
        model_type = st.selectbox("Select Regression Model", [
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Polynomial Regression",
            "Support Vector Regression (SVR)",
            "Decision Tree Regression",
            "Random Forest Regression"
        ])
    elif task == "Classification":
        model_type = st.selectbox("Select Classification Model", [
            "Logistic Regression",
            "K-Nearest Neighbors (K-NN)",
            "Support Vector Machine (SVM)",
            "Kernel SVM",
            "Naive Bayes",
            "Decision Tree Classification",
            "Random Forest Classification"
        ])
    elif task == "Clustering":
        model_type = st.selectbox("Select Clustering Model", [
            "K-Means Clustering",
            "Hierarchical Clustering"
        ])
        # ===== CLUSTERING SETTINGS =====
        n_clusters = None
        if task == "Clustering" and model_type in ["K-Means Clustering", "Hierarchical Clustering"]:
            n_clusters = st.number_input("Enter number of clusters", min_value=1, max_value=10, value=3)

    # ---------- MODEL TRAINING ----------
    if st.button("Train Model"):
        if X is None:
            st.error("Please select at least one independent variable before training.")
        elif task != "Clustering" and y is None:
            st.error("Please select both independent and dependent variables before training.")
        else:
            # If no train-test split, use all data
            if X_train is None:
                if task != "Clustering":
                    X_train, X_test, y_train, y_test = X, X, y, y
                else:
                    X_train, X_test = X, X  # y is not used for clustering

            # ===== REGRESSION =====
            if task == "Regression":
                if model_type == "Simple Linear Regression":
                    regressor = LinearRegression().fit(X_train, y_train)
                    predictions = regressor.predict(X_test)

                elif model_type == "Multiple Linear Regression":
                    regressor = LinearRegression().fit(X_train, y_train)
                    predictions = regressor.predict(X_test)

                    # ---------- Backward Elimination ----------
                    X_with_const = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
                    SL = 0.05
                    X_opt = X_with_const[:, list(range(X_with_const.shape[1]))]

                    while True:
                        X_opt = X_opt.astype(np.float64)
                        regressor_OLS = sm.OLS(y, X_opt).fit()
                        max_p_value = max(regressor_OLS.pvalues)
                        if max_p_value > SL:
                            max_p_index = np.argmax(regressor_OLS.pvalues)
                            X_opt = np.delete(X_opt, max_p_index, 1)
                        else:
                            break

                    st.subheader("Backward Elimination Summary")
                    st.text(regressor_OLS.summary())
                    st.write("Selected Features after Backward Elimination:", X_opt.shape[1])

                elif model_type == "Polynomial Regression":
                    poly = PolynomialFeatures(degree=4)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    regressor = LinearRegression().fit(X_train_poly, y_train)
                    predictions = regressor.predict(X_test_poly)

                elif model_type == "Support Vector Regression (SVR)":
                    regressor = SVR(kernel='rbf').fit(X_train, y_train)
                    predictions = regressor.predict(X_test)

                elif model_type == "Decision Tree Regression":
                    regressor = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
                    predictions = regressor.predict(X_test)

                elif model_type == "Random Forest Regression":
                    regressor = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_train, y_train)
                    predictions = regressor.predict(X_test)

                # ===== OPTIONAL DATA DISPLAY =====
                st.markdown("### Show Data Details")

                with st.expander("Dataset"):
                    st.dataframe(df)

                with st.expander("X (Features)"):
                    st.write(X)

                with st.expander("Y (Target)"):
                    st.write(y)

                with st.expander("Predictions"):
                    st.write(predictions)

                if (
                        not pd.DataFrame(X_train).equals(pd.DataFrame(X_test)) or
                        not pd.Series(y_train.ravel()).equals(pd.Series(y_test.ravel()))
                ):
                    with st.expander("X_train"):
                        st.write(X_train)
                    with st.expander("y_train"):
                        st.write(y_train)
                    with st.expander("X_test"):
                        st.write(X_test)
                    with st.expander("y_test"):
                        st.write(y_test)

                # ---------- MATPLOTLIB PLOT ----------
                if len(independent_vars) == 1:  # Only plot if single feature
                    plt.figure(figsize=(8, 5))
                    plt.scatter(X_train, y_train, color='red', label="Training Data")

                    if model_type == "Polynomial Regression":
                        X_grid = np.linspace(X_train.iloc[:, 0].min(), X_train.iloc[:, 0].max(), 300).reshape(-1, 1)
                        X_grid_poly = poly.transform(X_grid)
                        y_grid_pred = regressor.predict(X_grid_poly)
                        plt.plot(X_grid, y_grid_pred, color='blue', label="Polynomial Curve")
                    else:
                        plt.plot(X_train, regressor.predict(X_train), color='blue', label="Regression Line")

                    plt.xlabel(independent_vars[0])
                    plt.ylabel(dependent_vars)
                    plt.title(f"{model_type} - Training Data")
                    plt.legend()
                    st.pyplot(plt)
                else:
                    st.info("📊 Plotting is only available when you select **one independent variable**.")


            # ===== CLASSIFICATION =====
            elif task == "Classification":
                # --- Select model ---
                if model_type == "Logistic Regression":
                    model = LogisticRegression(random_state=0)
                elif model_type == "K-Nearest Neighbors (K-NN)":
                    model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                elif model_type == "Support Vector Machine (SVM)":
                    model = SVC(kernel='linear', random_state=0)
                elif model_type == "Kernel SVM":
                    model = SVC(kernel='rbf', random_state=0)
                elif model_type == "Naive Bayes":
                    model = GaussianNB()
                elif model_type == "Decision Tree Classification":
                    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
                elif model_type == "Random Forest Classification":
                    model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

                # --- Train and predict ---
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                st.write("Accuracy:", accuracy_score(y_test, predictions))

                st.markdown("### Show Data Details")
                with st.expander("Dataset"):
                    st.dataframe(df)

                with st.expander(" X (Features)"):
                    st.write(X)

                with st.expander(" Y (Target)"):
                    st.write(y)

                with st.expander("Predictions"):
                    st.write(predictions)

                if (
                        not pd.DataFrame(X_train).equals(pd.DataFrame(X_test)) or
                        not pd.Series(y_train.ravel()).equals(pd.Series(y_test.ravel()))
                ):
                    with st.expander(" X_train"):
                        st.write(X_train)

                    with st.expander(" y_train"):
                        st.write(y_train)

                    with st.expander("X_test"):
                        st.write(X_test)

                    with st.expander("y_test"):
                        st.write(y_test)

                # --- Optional Confusion Matrix ---
                cm = confusion_matrix(y_test, predictions)
                with st.expander("📊 Confusion Matrix"):
                    st.write(cm)

                # --- Plotting based on feature count ---
                num_features = X_train.shape[1] if len(X_train.shape) > 1 else 1

                if num_features == 1:
                    # ✅ Single feature scatter + prediction line
                    plt.figure(figsize=(8, 5))

                    X_train_arr, y_train_arr = np.array(X_train), np.array(y_train)
                    X_test_arr, y_test_arr = np.array(X_test), np.array(y_test)

                    plt.scatter(X_train_arr, y_train_arr, color='red', label="Training Data")
                    plt.scatter(X_test_arr, y_test_arr, color='green', label="Test Data")

                    # Sort X for smooth prediction line
                    X_sorted = np.sort(X_train_arr, axis=0)
                    y_pred_line = model.predict(X_sorted.reshape(-1, 1))
                    plt.plot(X_sorted, y_pred_line, color='blue', label="Prediction Line")

                    plt.xlabel(independent_vars[0])
                    plt.ylabel(dependent_vars)
                    plt.title(f"{model_type} - Classification (1 feature)")
                    plt.legend()
                    st.pyplot(plt)



                elif num_features == 2:

                    # Ensure numpy arrays

                    X_train, y_train = np.array(X_train), np.array(y_train)

                    X_test, y_test = np.array(X_test), np.array(y_test)

                    # Adaptive grid resolution

                    grid_points = 200 if len(X_train) < 5000 else 100

                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                    # --- TRAIN SET ---

                    x1, x2 = np.meshgrid(

                        np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, grid_points),
                        np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, grid_points)
                    )
                    Z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
                    axes[0].contourf(x1, x2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
                    for i, j in enumerate(np.unique(y_train)):
                        axes[0].scatter(
                            X_train[y_train == j, 0],
                            X_train[y_train == j, 1],
                            c=ListedColormap(('red', 'green'))(i), label=str(j)
                        )
                    axes[0].set_title(f"{model_type} - Training set")
                    axes[0].set_xlabel(independent_vars[0])
                    axes[0].set_ylabel(independent_vars[1])
                    axes[0].legend()

                    # --- TEST SET ---
                    x1, x2 = np.meshgrid(
                        np.linspace(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1, grid_points),
                        np.linspace(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1, grid_points)
                    )
                    Z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
                    axes[1].contourf(x1, x2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
                    for i, j in enumerate(np.unique(y_test)):
                        axes[1].scatter(
                            X_test[y_test == j, 0],
                            X_test[y_test == j, 1],
                            c=ListedColormap(('red', 'green'))(i), label=str(j)
                        )
                    axes[1].set_title(f"{model_type} - Test set")
                    axes[1].set_xlabel(independent_vars[0])
                    axes[1].set_ylabel(independent_vars[1])
                    axes[1].legend()
                    # Show combined plot
                    st.pyplot(fig)

            # ===== CLUSTERING =====
            if task == "Clustering":
                st.markdown("### Clustering Results")

                # Dependent variable is optional in clustering
                if dependent_vars != "None":
                    y_true = df[dependent_vars]
                else:
                    y_true = None

                # Elbow Method for K-Means
                if model_type == "K-Means Clustering":
                    wcss = []
                    for i in range(1, 11):
                        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                        km.fit(X)
                        wcss.append(km.inertia_)

                    # Plot Elbow
                    fig_elbow, ax_elbow = plt.subplots()
                    ax_elbow.plot(range(1, 11), wcss, marker='o')
                    ax_elbow.set_xlabel('Number of clusters')
                    ax_elbow.set_ylabel('WCSS')
                    ax_elbow.set_title('Elbow Method')
                    st.pyplot(fig_elbow)

                # Fit selected model
                if model_type == "K-Means Clustering":
                    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10,
                                   random_state=0)
                    predictions = model.fit_predict(X)
                else:
                    # Hierarchical Clustering
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    predictions = model.fit_predict(X)

                    # Dendrogram
                    linked = linkage(X, method='ward')
                    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
                    dendrogram(linked, truncate_mode='lastp', p=30)
                    ax_dendro.set_title('Hierarchical Clustering Dendrogram')
                    ax_dendro.set_xlabel('Sample Index')
                    ax_dendro.set_ylabel('Distance')
                    st.pyplot(fig_dendro)

                df["Prediction"] = predictions

                st.markdown("### Clustered Data")
                st.dataframe(df)

                with st.expander("X (Features)"):
                    st.write(X)
                with st.expander("Predictions"):
                    st.write(predictions)

                # 🔑 Evaluate clustering against target if available
                if y_true is not None:
                    from sklearn.metrics import adjusted_rand_score

                    ari = adjusted_rand_score(y_true, predictions)
                    st.success(f"Adjusted Rand Index (vs target {dependent_vars}): {ari:.3f}")

                # Labels and colors
                cluster_labels = ["Careful", "Standard", "Target", "Careless", "Sensible",
                                  "Cautious", "Adventurous", "Strategic", "Balanced", "Focused"]
                cluster_colors = ["red", "blue", "green", "cyan", "purple",
                                  "orange", "pink", "brown", "gray", "olive"]

                # Scatter plot
                if X.shape[1] >= 2:
                    fig, ax = plt.subplots()
                    for cluster_id in range(len(set(predictions))):
                        ax.scatter(
                            X.iloc[predictions == cluster_id, 0],
                            X.iloc[predictions == cluster_id, 1],
                            s=100,
                            c=cluster_colors[cluster_id],
                            label=cluster_labels[cluster_id]
                        )
                    # Plot centroids
                    if hasattr(model, "cluster_centers_"):
                        centroids = model.cluster_centers_
                    else:
                        centroids = np.array(
                            [X[predictions == cid].mean(axis=0) for cid in np.unique(predictions)])

                    ax.scatter(
                        centroids[:, 0],
                        centroids[:, 1],
                        s=300,
                        c='yellow',
                        marker='X',
                        edgecolor='black',
                        label='Centroids'
                    )
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.set_title(f"{model_type} Results")
                    ax.legend()
                    st.pyplot(fig)
