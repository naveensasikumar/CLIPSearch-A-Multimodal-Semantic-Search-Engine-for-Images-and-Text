import json
import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter

class SearchAnalytics:
    def __init__(self, data_file="../data/search_analytics.json"):
        self.data_file = data_file
        self.ensure_data_file()
    
    def ensure_data_file(self):
        if not os.path.exists(os.path.dirname(self.data_file)):
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump([], f)
    
    def log_search(self, query_type, original_query, processed_query, results_count, response_time, user_rating=None):
        search_event = {
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "original_query": original_query,
            "processed_query": processed_query,
            "results_count": results_count,
            "response_time_ms": response_time,
            "user_rating": user_rating,
            "session_id": st.session_state.get('session_id', 'unknown')
        }
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            data = []
        
        data.append(search_event)
        
        if len(data) > 1000:
            data = data[-1000:]
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
    
    def update_rating(self, search_index, rating):
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            if 0 <= search_index < len(data):
                data[search_index]["user_rating"] = rating
                
                with open(self.data_file, 'w') as f:
                    json.dump(data, f)
                return True
        except:
            pass
        return False
    
    def get_analytics_data(self, days=30):
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_data = [
            event for event in data 
            if datetime.fromisoformat(event['timestamp']) > cutoff_date
        ]
        
        return filtered_data
    
    def render_analytics_dashboard(self):
        st.markdown("## 📊 Search Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            days = st.selectbox("Time Period", [7, 14, 30, 90], index=2)
        with col2:
            auto_refresh = st.checkbox("Auto-refresh", value=False)
        
        if auto_refresh:
            st.rerun()
        
        data = self.get_analytics_data(days)
        
        if not data:
            st.info("No search data available yet. Start searching to see analytics!")
            return
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_searches = len(df)
            st.metric("Total Searches", total_searches)
        
        with col2:
            avg_response_time = df['response_time_ms'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms")
        
        with col3:
            text_searches = len(df[df['query_type'] == 'text'])
            st.metric("Text Searches", text_searches)
        
        with col4:
            rated_searches = df['user_rating'].notna().sum()
            if rated_searches > 0:
                avg_rating = df['user_rating'].mean()
                st.metric("Avg Rating", f"{avg_rating:.1f}⭐")
            else:
                st.metric("Avg Rating", "No ratings yet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily_searches = df.groupby('date').size().reset_index(name='count')
            fig = px.line(daily_searches, x='date', y='count', 
                         title='Daily Search Volume',
                         labels={'count': 'Number of Searches', 'date': 'Date'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            type_counts = df['query_type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title='Search Type Distribution')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        if len(df) > 10:
            hourly_activity = df.groupby(['date', 'hour']).size().reset_index(name='searches')
            if not hourly_activity.empty:
                pivot_data = hourly_activity.pivot(index='date', columns='hour', values='searches').fillna(0)
                
                fig = px.imshow(pivot_data.values,
                               labels=dict(x="Hour of Day", y="Date", color="Searches"),
                               x=[f"{h:02d}:00" for h in range(24)],
                               y=pivot_data.index,
                               title="Search Activity Heatmap",
                               color_continuous_scale="Blues")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔥 Popular Queries")
        if 'processed_query' in df.columns:
            popular_queries = df['processed_query'].value_counts().head(10)
            if not popular_queries.empty:
                fig = px.bar(x=popular_queries.values, y=popular_queries.index,
                           orientation='h', title='Top 10 Search Queries')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        if len(df) > 5:
            st.markdown("### ⚡ Performance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='response_time_ms', nbins=20,
                                 title='Response Time Distribution')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if df['query_type'].nunique() > 1:
                    fig = px.box(df, x='query_type', y='response_time_ms',
                               title='Response Time by Search Type')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📝 Recent Searches")
        recent_df = df.tail(10)[['timestamp', 'query_type', 'processed_query', 'results_count', 'response_time_ms', 'user_rating']]
        recent_df = recent_df.sort_values('timestamp', ascending=False)
        st.dataframe(recent_df, use_container_width=True)

analytics = SearchAnalytics()