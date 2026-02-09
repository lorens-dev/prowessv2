import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & CSS INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inbox", layout="wide", page_icon="📊")

def inject_custom_css():
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp { background-color: #f3f4f6; }

        /* CARD STYLING */
        .dashboard-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 20px;
            border: 1px solid #e5e7eb;
        }

        /* METRIC CARDS */
        .metric-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .metric-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 10px;
            font-size: 20px;
        }
        .icon-blue { background-color: #dbeafe; color: #2563eb; }
        .icon-green { background-color: #dcfce7; color: #16a34a; }
        .icon-purple { background-color: #f3e8ff; color: #9333ea; }
        .icon-orange { background-color: #ffedd5; color: #ea580c; }
        .icon-red { background-color: #fee2e2; color: #dc2626; }
        .icon-gray { background-color: #f3f4f6; color: #4b5563; }

        .metric-data { text-align: right; }
        .metric-label {
            font-size: 0.85rem; color: #6b7280; font-weight: 500;
            text-transform: uppercase; letter-spacing: 0.05em;
        }
        .metric-value { font-size: 1.5rem; font-weight: 700; color: #111827; }
        .metric-delta { font-size: 0.75rem; font-weight: 600; margin-top: 4px; }
        .delta-pos { color: #16a34a; }
        .delta-neg { color: #dc2626; }
        .delta-neu { color: #6b7280; }

        .sidebar-header {
            font-size: 1.1rem; font-weight: 700; color: #374151;
            margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def clean_currency(x):
    if isinstance(x, str):
        clean_str = x.replace('₱', '').replace('$', '').replace(',', '').strip()
        try: return float(clean_str)
        except ValueError: return 0.0
    return x

# -----------------------------------------------------------------------------
# 2. ETL PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_and_filter_data(start_h, end_h):
    # --- FIXED: Use st.connection to automatically read secrets.toml ---
    try:
        # "mydb" matches [connections.mydb] in your secrets file
        conn = st.connection("mydb", type="sql")
        
        # Use conn.query() - simpler and automatically caches
        df_trans = conn.query("SELECT * FROM transactions", ttl=600)
        df_td = conn.query("SELECT * FROM transactiondetails", ttl=600)
        df_prod = conn.query("SELECT * FROM products", ttl=600)
    except Exception as e:
        st.error(f"Database Connection Error: {str(e)}")
        st.stop()

    # Normalization
    df_trans.columns = df_trans.columns.str.lower()
    df_td.columns = df_td.columns.str.lower()
    df_prod.columns = df_prod.columns.str.lower()

    rename_map = {'product_id': 'productid', 'transaction_id': 'tid'}
    df_td.rename(columns=rename_map, inplace=True)
    df_prod.rename(columns=rename_map, inplace=True)
    df_trans.rename(columns=rename_map, inplace=True)

    df_trans['tdate'] = pd.to_datetime(df_trans['tdate'], errors='coerce')
    df_trans.dropna(subset=['tdate'], inplace=True)

    # Clean Currency Columns
    for col in ['gross', 'cogs', 'grossprofit']:
        if col in df_trans.columns:
            if df_trans[col].dtype == 'object':
                df_trans[col] = df_trans[col].apply(clean_currency)
            df_trans[col] = pd.to_numeric(df_trans[col], errors='coerce').fillna(0)

    # Filter Voids
    if 'isvoid' in df_trans.columns:
        df_trans['isvoid'] = pd.to_numeric(df_trans['isvoid'], errors='coerce').fillna(0)
        df_trans = df_trans[df_trans['isvoid'] == 0].copy()

    # Time Features
    df_trans['hour'] = df_trans['tdate'].dt.hour
    df_trans['date_only'] = df_trans['tdate'].dt.date
    
    # Filter by Shift Hours
    if start_h <= end_h:
        df_trans = df_trans[(df_trans['hour'] >= start_h) & (df_trans['hour'] <= end_h)]
    else:
        # Overnight shift (e.g., 10 PM to 2 AM)
        df_trans = df_trans[(df_trans['hour'] >= start_h) | (df_trans['hour'] <= end_h)]

    # Clean Product IDs
    if 'productid' in df_td.columns: 
        df_td['productid'] = df_td['productid'].astype(str).str.replace(r'\.0$', '', regex=True)
    if 'productid' in df_prod.columns: 
        df_prod['productid'] = df_prod['productid'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Merge Data
    if 'productid' in df_td.columns and 'productid' in df_prod.columns:
        df_full = df_td.merge(df_prod, on='productid', how='left')
    else:
        df_full = df_td 

    if 'tid' in df_trans.columns and 'tid' in df_full.columns:
        df_trans['tid'] = df_trans['tid'].astype(str)
        df_full['tid'] = df_full['tid'].astype(str)
        valid_tids = df_trans['tid'].unique()
        df_full = df_full[df_full['tid'].isin(valid_tids)]
        df_full = df_full.merge(df_trans[['tid', 'date_only']], on='tid', how='inner')
    
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]

    if 'quantity' in df_full.columns:
        df_full['quantity'] = pd.to_numeric(df_full['quantity'], errors='coerce').fillna(0).abs()

    if 'sellingprice' in df_full.columns:
        if df_full['sellingprice'].dtype == 'object':
            df_full['sellingprice'] = df_full['sellingprice'].apply(clean_currency)
        df_full['sellingprice'] = pd.to_numeric(df_full['sellingprice'], errors='coerce').fillna(0)

    # Standardize Product Name
    potential_names = ['description_x', 'description', 'productname', 'name', 'item_name', 'product_name']
    found_col = None
    for cand in potential_names:
        if cand in df_full.columns:
            found_col = cand
            break
            
    if found_col:
        if found_col != 'productname':
            if 'productname' in df_full.columns:
                df_full = df_full.drop(columns=['productname'])
            df_full.rename(columns={found_col: 'productname'}, inplace=True)
    else:
        if 'productname' not in df_full.columns:
            df_full['productname'] = "Unknown Product"

    return df_trans, df_full

# -----------------------------------------------------------------------------
# 3. FORECASTING & LOGIC
# -----------------------------------------------------------------------------
def train_model(df_trans):
    daily_sales = df_trans.groupby('date_only')['gross'].sum().reset_index()
    daily_sales['date_only'] = pd.to_datetime(daily_sales['date_only'])

    if daily_sales.empty:
        return None, pd.DataFrame()

    full_idx = pd.date_range(start=daily_sales['date_only'].min(), end=daily_sales['date_only'].max(), freq='D')
    daily_sales = daily_sales.set_index('date_only').reindex(full_idx, fill_value=0).reset_index()
    daily_sales.rename(columns={'index': 'date_only'}, inplace=True)

    daily_sales['day_of_week'] = daily_sales['date_only'].dt.dayofweek
    daily_sales['is_weekend'] = daily_sales['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    daily_sales['month'] = daily_sales['date_only'].dt.month
    daily_sales['lag_1'] = daily_sales['gross'].shift(1)
    daily_sales['lag_7'] = daily_sales['gross'].shift(7)
    daily_sales['rolling_mean_7'] = daily_sales['gross'].rolling(window=7).mean()
    daily_sales.dropna(inplace=True)

    if len(daily_sales) < 10:
        return None, daily_sales

    X = daily_sales[['day_of_week', 'is_weekend', 'month', 'lag_1', 'lag_7', 'rolling_mean_7']]
    y = daily_sales['gross']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, daily_sales

def get_prediction(model, recent_data, target_date):
    if model is None or recent_data.empty: return 0.0
    target_dt = pd.to_datetime(target_date)
    last_row = recent_data.iloc[-1]
    
    # Safely get lag values
    lag_7_val = recent_data.iloc[-7]['gross'] if len(recent_data) >= 7 else last_row['gross']
    
    features = pd.DataFrame([[
        target_dt.dayofweek, 
        1 if target_dt.dayofweek >= 5 else 0, 
        target_dt.month, 
        last_row['gross'], 
        lag_7_val, 
        recent_data['gross'].tail(7).mean()
    ]], columns=['day_of_week', 'is_weekend', 'month', 'lag_1', 'lag_7', 'rolling_mean_7'])
    
    return max(0, model.predict(features)[0])

def get_weekly_stock_recommendations(df_full, start_date):
    if 'productname' not in df_full.columns: return pd.DataFrame()

    df_full['date_only'] = pd.to_datetime(df_full['date_only'])
    dataset_end_date = df_full['date_only'].max()

    stats = df_full.groupby('productname').agg(
        total_qty=('quantity', 'sum'),
        avg_price=('sellingprice', 'mean'),
        first_sale_date=('date_only', 'min') 
    ).reset_index()
    
    stats['days_on_menu'] = (dataset_end_date - stats['first_sale_date']).dt.days + 1
    stats['days_on_menu'] = stats['days_on_menu'].clip(lower=1)
    stats['velocity'] = stats['total_qty'] / stats['days_on_menu'] 

    recommendations = []
    
    for _, row in stats.iterrows():
        weekly_demand = 0
        for i in range(7):
            current_day = pd.to_datetime(start_date) + timedelta(days=i)
            multiplier = 1.3 if current_day.weekday() >= 4 else 1.0 
            weekly_demand += row['velocity'] * multiplier

        final_weekly_demand = int(np.ceil(weekly_demand))
        
        # Velocity Ranking Logic
        velocity_score = 0
        if row['velocity'] > 10: velocity_score += 40
        elif row['velocity'] > 5: velocity_score += 30
        else: velocity_score += 5
        
        if final_weekly_demand > 50: velocity_score += 30
        elif final_weekly_demand > 30: velocity_score += 20
        
        if row['avg_price'] > 200: velocity_score += 20
        elif row['avg_price'] > 100: velocity_score += 10
        
        if row['days_on_menu'] > 14: velocity_score += 10
        
        if velocity_score >= 70: priority = "High"
        elif velocity_score >= 40: priority = "Medium"
        else: priority = "Low"

        if final_weekly_demand > 0:
            recommendations.append({
                "Product": row['productname'],
                "Weekly Demand": final_weekly_demand,
                "Priority": priority,
                "Velocity Score": velocity_score,
                "Avg/Shift": round(row['velocity'], 2), 
                "Price": row['avg_price']
            })
    
    df_rec = pd.DataFrame(recommendations)
    if not df_rec.empty:
        df_rec = df_rec.sort_values(by=["Velocity Score"], ascending=False)
        
    return df_rec

def get_marketing_insights(daily_sales, target_date):
    if daily_sales.empty or len(daily_sales) < 14:
        return "Insufficient Data", "Need more history.", []

    daily_sales['date_only'] = pd.to_datetime(daily_sales['date_only'])
    target_dt = pd.to_datetime(target_date)
    past_data = daily_sales[daily_sales['date_only'] <= target_dt].sort_values('date_only')
    
    if len(past_data) < 14: return "Insufficient Data", "History too sparse.", []

    curr_rev = past_data.iloc[-7:]['gross'].sum()
    prev_rev = past_data.iloc[-14:-7]['gross'].sum()
    growth = ((curr_rev - prev_rev) / prev_rev) * 100 if prev_rev != 0 else 0

    if growth < -10:
        return "Revenue Alert", "Recent volume is significantly lower than average.", ["Launch BOGO Deal", "SMS Blast", "Coupons"]
    elif growth < 0:
        return "Slight Dip", "Performance is slightly below the baseline.", ["Lunch Combos", "Social Push", "Ask Reviews"]
    elif growth < 10:
        return "Steady", "Revenue flow is consistent and stable.", ["Loyalty Card", "Upselling", "Happy Hour"]
    else:
        return "High Performance", "Strong positive trend detected.", ["Premium Specials", "Referrals", "Gift Cards"]

# -----------------------------------------------------------------------------
# 4. CUSTOM COMPONENT FUNCTIONS
# -----------------------------------------------------------------------------
def display_metric_card(title, value, icon_class, color_class, delta_text="", delta_color_class="delta-neu"):
    html = f"""
    <div class="dashboard-card metric-container">
        <div class="metric-icon {color_class}">
            <i class="{icon_class}"></i>
        </div>
        <div class="metric-data">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_color_class}">{delta_text}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MAIN APP
# -----------------------------------------------------------------------------
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        try:
            st.image("logo.png", use_container_width=True)
        except:
            st.markdown("## <i class='fas fa-chart-pie'></i> Dashboard", unsafe_allow_html=True)
        
        st.markdown("<div class='sidebar-header'><i class='fas fa-sliders'></i> Controls</div>", unsafe_allow_html=True)
        
        today = datetime.now().date()
        target_date = st.date_input("Forecast Date", today)
        
        st.markdown("---")
        st.markdown("<div class='sidebar-header'><i class='far fa-clock'></i> Shift (12H)</div>", unsafe_allow_html=True)
        t_start = st.time_input("Start", time(17, 0)) 
        t_end = st.time_input("End", time(22, 0))

        st.info("Filtering by specific shift hours.")

    # --- PROCESS DATA ---
    try:
        df_trans, df_full = load_and_filter_data(t_start.hour, t_end.hour)
        model, recent_data = train_model(df_trans)
    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        st.stop()

    # --- CALCULATIONS ---
    # Ensure date types match for comparison
    df_trans['date_only'] = pd.to_datetime(df_trans['date_only']).dt.date
    actual_day_data = df_trans[df_trans['date_only'] == target_date]
    has_actual_data = not actual_day_data.empty

    if has_actual_data:
        metric_rev = actual_day_data['gross'].sum()
        metric_orders = len(actual_day_data)
        rev_label, ord_label = "Actual Revenue", "Actual Orders"
        rev_delta_text, rev_delta_color = "Live Data", "delta-pos"
    else:
        metric_rev = get_prediction(model, recent_data, target_date)
        avg_ticket = df_trans['gross'].mean() if not df_trans.empty else 0
        metric_orders = int(metric_rev / avg_ticket) if avg_ticket > 0 else 0
        rev_label, ord_label = "Predicted Revenue", "Predicted Orders"
        rev_delta_text, rev_delta_color = "AI Forecast", "delta-neu"

    monthly_forecast = 0
    if model is not None and not df_trans.empty:
        # Convert target_date to datetime if it's date
        target_dt = pd.to_datetime(target_date)
        m_start = target_dt.replace(day=1)
        m_end = (m_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        # Ensure comparison works
        month_mask = (pd.to_datetime(df_trans['date_only']) >= m_start) & (pd.to_datetime(df_trans['date_only']) <= m_end)
        actuals = df_trans.loc[month_mask, 'gross'].sum()
        
        dates_present = set(pd.to_datetime(df_trans.loc[month_mask, 'date_only']))
        
        future_sum = 0
        curr = m_start
        while curr <= m_end:
            if curr not in dates_present: 
                future_sum += get_prediction(model, recent_data, curr)
            curr += timedelta(days=1)
        monthly_forecast = actuals + future_sum
    
    # --- VS YESTERDAY ---
    yesterday_date = target_date - timedelta(days=1)
    yesterday_mask = df_trans['date_only'] == yesterday_date
    yesterday_rev = df_trans.loc[yesterday_mask, 'gross'].sum()
    
    pct_diff = 0
    if yesterday_rev > 0:
        pct_diff = ((metric_rev - yesterday_rev) / yesterday_rev) * 100
    elif yesterday_rev == 0 and metric_rev > 0:
        pct_diff = 100.0
    
    vs_label = f"{abs(pct_diff):.1f}%"
    if pct_diff > 0:
        vs_icon, vs_color, vs_delta_class, vs_text = "fa-arrow-trend-up", "icon-green", "delta-pos", "Increase"
    elif pct_diff < 0:
        vs_icon, vs_color, vs_delta_class, vs_text = "fa-arrow-trend-down", "icon-red", "delta-neg", "Decrease"
    else:
        vs_icon, vs_color, vs_delta_class, vs_text = "fa-minus", "icon-gray", "delta-neu", "No Change"

    # Demand Calc
    top_movers, high_velocity_items = [], []
    avg_velocity_score = 0
    if not df_full.empty:
        weekly_inv = get_weekly_stock_recommendations(df_full, target_date)
        if not weekly_inv.empty:
            top_movers = weekly_inv[weekly_inv['Priority'] == 'Medium']['Product'].tolist()
            high_velocity_items = weekly_inv[weekly_inv['Priority'] == 'High']['Product'].tolist()
            if high_velocity_items: 
                avg_velocity_score = weekly_inv[weekly_inv['Priority'] == 'High']['Velocity Score'].mean()

    # --- UI: HEADER ---
    st.markdown(f"### <i class='fas fa-tachometer-alt'></i> Performance: {target_date.strftime('%b %d, %Y %A')}", unsafe_allow_html=True)

    # --- UI: METRIC CARDS ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        display_metric_card(rev_label, f"₱{metric_rev:,.0f}", "fas fa-coins", "icon-blue", rev_delta_text, rev_delta_color)
    with c2:
        display_metric_card(ord_label, f"{metric_orders}", "fas fa-shopping-basket", "icon-purple", "Transactions", "delta-neu")
    with c3:
        display_metric_card("Month Forecast", f"₱{monthly_forecast:,.0f}", "fas fa-calendar-alt", "icon-green", target_date.strftime('%B'), "delta-neu")
    with c4:
        display_metric_card("Top Movers", str(len(top_movers)), "fas fa-fire", "icon-orange", "Active Demand", "delta-pos")
    with c5:
        display_metric_card("Velocity Score", f"{avg_velocity_score:.0f}%", "fas fa-bolt", "icon-red", f"{len(high_velocity_items)} High Traffic", "delta-pos")
    with c6:
        display_metric_card("Vs Yesterday", vs_label, f"fas {vs_icon}", vs_color, vs_text, vs_delta_class)

    # --- UI: MAIN TABS ---
    t1, t2, t3 = st.tabs(["Revenue Intelligence", "Demand Forecast", "Marketing AI"])

    with t1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        col_chart, col_log = st.columns([2, 1])
        with col_chart:
            st.subheader("Revenue Trend")
            if not recent_data.empty:
                fig = px.line(recent_data, x='date_only', y='gross', template="plotly_white")
                fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
                fig.add_trace(go.Scatter(
                    x=[target_date], y=[metric_rev], mode='markers+text',
                    name=rev_label, marker=dict(color='#ef4444' if not has_actual_data else '#22c55e', size=12),
                    text=[f"₱{metric_rev:,.0f}"], textposition="top center"
                ))
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("No data.")

        with col_log:
            st.subheader("Shift Log")
            if has_actual_data:
                cols = [c for c in ['tdate', 'gross', 'paymentmethod'] if c in actual_day_data.columns]
                st.dataframe(actual_day_data[cols].style.format({'gross': '₱{:.2f}'}), use_container_width=True, height=300)
            else:
                st.info("No live transactions yet. Using AI prediction.")
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        
        # Calculate End Date for the 7-day display
        end_date = target_date + timedelta(days=6)
        st.subheader(f"Demand Plan: {target_date.strftime('%b %d')} - {end_date.strftime('%b %d')}")
        
        if not df_full.empty and not weekly_inv.empty:
            def highlight_priority(val):
                if val == 'High': return 'background-color: #fee2e2; color: #991b1b; font-weight: bold;'
                elif val == 'Medium': return 'background-color: #ffedd5; color: #9a3412; font-weight: bold;'
                return 'background-color: #dcfce7; color: #166534;'

            st.dataframe(
                weekly_inv.style.map(highlight_priority, subset=['Priority'])
                        .format({'Velocity Score': '{:.0f}%', 'Price': '₱{:.0f}'}),
                use_container_width=True, 
                height=400
            )
        else:
            st.info("No sales data found to generate forecast. Check your database connection.")
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        if not recent_data.empty:
            m_title, m_desc, m_strats = get_marketing_insights(recent_data, target_date)
            st.subheader(f"{m_title}")
            st.write(m_desc)
            st.divider()
            st.markdown("#### Recommended Actions")
            for strat in m_strats:
                st.markdown(f"""
                <div style="background: #f9fafb; padding: 10px 15px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #3b82f6;">
                    <i class="fas fa-check-circle" style="color: #3b82f6; margin-right: 8px;"></i> {strat}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient data.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
