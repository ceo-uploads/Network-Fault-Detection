import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Shopping Assistant", layout="centered")

st.title("🛒 Smart Retail Product Recommender")
st.write(
    "Discover what products customers usually pick together based on past shopping habits."
)


@st.cache_data
def fetch_shopping_rules():
    return joblib.load("mba_rules.pkl")


try:
    shopping_rules = fetch_shopping_rules()
except Exception:
    st.error(
        "Shopping patterns file not found! Please run 'python train_mba.py' first."
    )
    st.stop()

available_products = [
    "Milk",
    "Bread",
    "Butter",
    "Diaper",
    "Beer",
    "Coffee",
    "Eggs",
    "Cheese",
]

st.subheader("Select items currently in the customer's cart:")
current_cart = st.multiselect(
    "Customer Cart:", available_products, default=["Milk"]
)

if st.button("Find Smart Recommendations"):
    if not current_cart:
        st.warning(
            "Please pick at least one product to see smart recommendations."
        )
    else:
        matching_suggestions = []
        for _, rule in shopping_rules.iterrows():
            items_needed = set(rule["cart_items"])
            suggested_product = set(rule["suggested_items"])

            if items_needed.issubset(set(current_cart)) and not suggested_product.issubset(
                set(current_cart)
            ):
                matching_suggestions.append(
                    {
                        "Recommended Product": ", ".join(suggested_product),
                        "Likelihood (%)": round(rule["confidence"] * 100, 1),
                        "Rule Strength (Lift)": round(rule["lift"], 2),
                        "Popularity (%)": round(rule["support"] * 100, 1),
                    }
                )

        if matching_suggestions:
            results_df = (
                pd.DataFrame(matching_suggestions)
                .sort_values(by="Likelihood (%)", ascending=False)
                .drop_duplicates(subset=["Recommended Product"])
            )

            st.markdown("---")
            st.subheader("💡 Best Add-On Suggestions:")
            st.dataframe(results_df, use_container_width=True)

            top_product = results_df.iloc[0]["Recommended Product"]
            st.success(
                f"🌟 **Top Recommendation:** Ask customer if they would like to add **{top_product}** to their order!"
            )
        else:
            st.info(
                "No strong add-on suggestions found for this specific combination of items."
            )