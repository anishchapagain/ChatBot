"""
Module for the data analysis dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from modules.data_loader import DataLoader
from modules.data_analyzer import DataAnalyzer
from modules.error_handler import ErrorHandler

class Dashboard:
    """
    Dashboard for data visualization and analysis.
    """
    def __init__(
        self, 
        data_loader: DataLoader, 
        error_handler: ErrorHandler,
        config: Dict
    ):
        """
        Initialize the dashboard.
        
        Args:
            data_loader: Data loader instance
            error_handler: Error handler instance
            config: Application configuration
        """
        self.data_loader = data_loader
        self.error_handler = error_handler
        self.config = config
        self.analyzer = DataAnalyzer(error_handler)
        self.chart_height = config.get("ui", {}).get("chart_height", 400)
    
    def render(self):
        """Render the dashboard interface."""
        try:
            st.header("Data Analysis Dashboard")
            
            # Get current data
            df = self.data_loader.get_data()
            
            if df is None:
                st.info("No data loaded. Please upload a file in the Chat Interface tab.")
                return
            
            # Create sidebar for dashboard controls
            st.sidebar.header("Dashboard Controls")
            
            # Analysis type selector
            analysis_type = st.sidebar.selectbox(
                "Select Analysis",
                ["Data Overview", "Summary Statistics", "Distribution Analysis", 
                 "Correlation Analysis", "Time Series Analysis", "Custom Analysis"]
            )
            
            # Render the selected analysis
            if analysis_type == "Data Overview":
                self._render_data_overview(df)
            elif analysis_type == "Summary Statistics":
                self._render_summary_statistics(df)
            elif analysis_type == "Distribution Analysis":
                self._render_distribution_analysis(df)
            elif analysis_type == "Correlation Analysis":
                self._render_correlation_analysis(df)
            elif analysis_type == "Time Series Analysis":
                self._render_time_series_analysis(df)
            elif analysis_type == "Custom Analysis":
                self._render_custom_analysis(df)
        
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "Error rendering dashboard")
            st.error(error_msg)
    
    def _render_data_overview(self, df: pd.DataFrame):
        """
        Render data overview section.
        
        Args:
            df: DataFrame to analyze
        """
        st.subheader("Data Overview")
        
        # Basic information
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column information
        st.subheader("Column Information")
        
        # Prepare column info
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = missing / len(df) * 100
            unique_values = df[col].nunique()
            
            column_info.append({
                "Column Name": col,
                "Data Type": dtype,
                "Missing Values": missing,
                "Missing (%)": f"{missing_pct:.2f}%",
                "Unique Values": unique_values
            })
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df)
        
        # Data sample
        with st.expander("Data Sample"):
            st.dataframe(df.head())
        
        # Missing values chart
        missing_data = df.isna().sum()
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            st.subheader("Missing Values Chart")
            fig = px.bar(
                x=missing_data.index, 
                y=missing_data.values,
                labels={"x": "Column", "y": "Missing Values"},
                title="Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_summary_statistics(self, df: pd.DataFrame):
        """
        Render summary statistics section.
        
        Args:
            df: DataFrame to analyze
        """
        st.subheader("Summary Statistics")
        
        # Get numerical columns
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
        
        if not numerical_columns.empty:
            # Calculate summary statistics
            summary = df[numerical_columns].describe().T
            
            # Add additional statistics
            summary["range"] = summary["max"] - summary["min"]
            summary["iqr"] = summary["75%"] - summary["25%"]
            summary["mad"] = df[numerical_columns].mad().values
            summary["missing"] = df[numerical_columns].isna().sum().values
            summary["missing_pct"] = (summary["missing"] / len(df) * 100).round(2)
            
            # Format the display
            st.dataframe(summary)
            
            # Allow the user to select a column for visualization
            selected_column = st.selectbox(
                "Select a column for detailed statistics",
                numerical_columns
            )
            
            if selected_column:
                col_data = df[selected_column].dropna()
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                # Histogram
                with col1:
                    fig = px.histogram(
                        df, 
                        x=selected_column,
                        title=f"Distribution of {selected_column}",
                        height=self.chart_height
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                with col2:
                    fig = px.box(
                        df, 
                        y=selected_column,
                        title=f"Box Plot of {selected_column}",
                        height=self.chart_height
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical columns found in the dataset.")
            
        # Categorical columns analysis
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns
        
        if not categorical_columns.empty:
            st.subheader("Categorical Data Analysis")
            
            selected_cat_column = st.selectbox(
                "Select a categorical column for analysis",
                categorical_columns
            )
            
            if selected_cat_column:
                # Value counts
                value_counts = df[selected_cat_column].value_counts()
                
                # Limit the number of categories shown
                max_categories = st.slider("Maximum categories to display", 5, 20, 10)
                
                if len(value_counts) > max_categories:
                    # Keep top categories and group others
                    top_counts = value_counts.iloc[:max_categories-1]
                    other_count = value_counts.iloc[max_categories-1:].sum()
                    
                    # Create "Other" category
                    value_counts = pd.concat([top_counts, pd.Series({"Other": other_count})])
                
                # Create bar chart
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    labels={"x": selected_cat_column, "y": "Count"},
                    title=f"Value Counts for {selected_cat_column}",
                    height=self.chart_height
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display value counts as table
                st.write("Value Counts:")
                value_counts_df = pd.DataFrame({
                    "Value": value_counts.index,
                    "Count": value_counts.values,
                    "Percentage": (value_counts.values / len(df) * 100).round(2)
                })
                st.dataframe(value_counts_df)
    
    def _render_distribution_analysis(self, df: pd.DataFrame):
        """
        Render distribution analysis section.
        
        Args:
            df: DataFrame to analyze
        """
        st.subheader("Distribution Analysis")
        
        # Get numerical columns
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
        
        if numerical_columns.empty:
            st.info("No numerical columns found in the dataset.")
            return
        
        # Create distribution plots
        plot_data = self.analyzer.create_distribution_plots(df)
        
        if plot_data:
            # Allow user to select columns to display
            selected_columns = st.multiselect(
                "Select columns to display",
                [plot["column"] for plot in plot_data],
                default=[plot["column"] for plot in plot_data[:min(3, len(plot_data))]]
            )
            
            # Filter selected plots
            selected_plots = [plot for plot in plot_data if plot["column"] in selected_columns]
            
            # Display plots
            for plot in selected_plots:
                column = plot["column"]
                st.subheader(f"Distribution of {column}")
                
                fig = go.Figure()
                
                # Add histogram
                fig.add_trace(go.Bar(
                    x=plot["x"],
                    y=plot["y"],
                    name="Frequency"
                ))
                
                # Add mean and median lines
                fig.add_vline(
                    x=plot["mean"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {plot['mean']:.2f}",
                    annotation_position="top right"
                )
                
                fig.add_vline(
                    x=plot["median"],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Median: {plot['median']:.2f}",
                    annotation_position="top left"
                )
                
                fig.update_layout(
                    title=f"Distribution of {column}",
                    xaxis_title=column,
                    yaxis_title="Frequency",
                    height=self.chart_height
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats for this column
                col_data = df[column].dropna()
                stats = col_data.describe()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{stats['mean']:.2f}")
                col2.metric("Median", f"{stats['50%']:.2f}")
                col3.metric("Std Dev", f"{stats['std']:.2f}")
                col4.metric("IQR", f"{stats['75%'] - stats['25%']:.2f}")
        else:
            st.info("Could not create distribution plots for this dataset.")
    
    def _render_correlation_analysis(self, df: pd.DataFrame):
        """
        Render correlation analysis section.
        
        Args:
            df: DataFrame to analyze
        """
        st.subheader("Correlation Analysis")
        
        # Get numerical columns
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
        
        if len(numerical_columns) < 2:
            st.info("At least two numerical columns are required for correlation analysis.")
            return
        
        # Create correlation matrix
        corr_data = self.analyzer.create_correlation_matrix(df)
        
        if corr_data:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_data["z"],
                x=corr_data["x"],
                y=corr_data["y"],
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=[[f"{val:.2f}" for val in row] for row in corr_data["z"]],
                texttemplate="%{text}",
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                height=max(self.chart_height, 500)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Allow user to select two columns for scatter plot
            st.subheader("Correlation Scatter Plot")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_column = st.selectbox("Select X-axis column", numerical_columns, index=0)
            
            with col2:
                # Default to second column if available
                default_index = min(1, len(numerical_columns) - 1)
                y_column = st.selectbox("Select Y-axis column", numerical_columns, index=default_index)
            
            if x_column and y_column:
                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    trendline="ols",
                    title=f"Correlation between {x_column} and {y_column}",
                    height=self.chart_height
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation coefficient
                corr_coef = df[x_column].corr(df[y_column])
                st.metric("Pearson Correlation Coefficient", f"{corr_coef:.4f}")
                
                # Add interpretation
                st.markdown("**Interpretation:**")
                if abs(corr_coef) < 0.3:
                    st.write("Weak correlation")
                elif abs(corr_coef) < 0.7:
                    st.write("Moderate correlation")
                else:
                    st.write("Strong correlation")
                
                if corr_coef > 0:
                    st.write("Positive correlation: As one variable increases, the other tends to increase as well.")
                else:
                    st.write("Negative correlation: As one variable increases, the other tends to decrease.")
        else:
            st.info("Could not create correlation matrix for this dataset.")
    
    def _render_time_series_analysis(self, df: pd.DataFrame):
        """
        Render time series analysis section.
        
        Args:
            df: DataFrame to analyze
        """
        st.subheader("Time Series Analysis")
        
        # Detect possible date columns
        date_columns = self.analyzer.detect_possible_date_columns(df)
        
        if not date_columns:
            st.info("No date columns detected in the dataset.")
            return
        
        # Get numerical columns
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
        
        if numerical_columns.empty:
            st.info("No numerical columns found for time series analysis.")
            return
        
        # Allow user to select columns
        col1, col2 = st.columns(2)
        
        with col1:
            selected_date_column = st.selectbox("Select date column", date_columns)
        
        with col2:
            selected_value_column = st.selectbox("Select value column", numerical_columns)
        
        if selected_date_column and selected_value_column:
            # Create time series plot
            plot_data = self.analyzer.create_time_series_plot(
                df, selected_date_column, selected_value_column
            )
            
            if plot_data:
                fig = go.Figure(data=go.Scatter(
                    x=plot_data["x"],
                    y=plot_data["y"],
                    mode='lines+markers'
                ))
                
                fig.update_layout(
                    title=f"{selected_value_column} over time",
                    xaxis_title=selected_date_column,
                    yaxis_title=selected_value_column,
                    height=self.chart_height
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series analysis options
                st.subheader("Time Series Analysis Options")
                
                analysis_type = st.selectbox(
                    "Select analysis type",
                    ["Raw Data", "Moving Average", "Year-over-Year Comparison", "Monthly Aggregation"]
                )
                
                # Process based on analysis type
                if analysis_type == "Raw Data":
                    # Already displayed above
                    pass
                
                elif analysis_type == "Moving Average":
                    # Add moving average to the plot
                    window_size = st.slider("Window size", 2, 30, 7)
                    
                    # Convert to pandas Series for rolling calculation
                    try:
                        # Handle date conversion if needed
                        if not pd.api.types.is_datetime64_any_dtype(df[selected_date_column]):
                            date_series = pd.to_datetime(df[selected_date_column], errors="coerce")
                        else:
                            date_series = df[selected_date_column]
                            
                        # Create a temporary dataframe with valid dates and values
                        temp_df = pd.DataFrame({
                            'date': date_series,
                            'value': df[selected_value_column]
                        }).dropna()
                        
                        # Sort by date
                        temp_df = temp_df.sort_values('date')
                        
                        # Calculate moving average
                        temp_df['moving_avg'] = temp_df['value'].rolling(window=window_size).mean()
                        
                        # Create figure
                        fig = go.Figure()
                        
                        # Add raw data
                        fig.add_trace(go.Scatter(
                            x=temp_df['date'],
                            y=temp_df['value'],
                            mode='lines',
                            name='Raw Data'
                        ))
                        
                        # Add moving average
                        fig.add_trace(go.Scatter(
                            x=temp_df['date'],
                            y=temp_df['moving_avg'],
                            mode='lines',
                            name=f'{window_size}-Period Moving Average',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_value_column} with {window_size}-Period Moving Average",
                            xaxis_title=selected_date_column,
                            yaxis_title=selected_value_column,
                            height=self.chart_height
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        self.error_handler.handle_error(e, "Error calculating moving average")
                        st.error("Could not calculate moving average. Please check your data.")
                
                elif analysis_type == "Year-over-Year Comparison":
                    try:
                        # Convert to datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(df[selected_date_column]):
                            df = df.copy()
                            df[selected_date_column] = pd.to_datetime(df[selected_date_column], errors="coerce")
                        
                        # Create a copy with valid dates and values
                        temp_df = df[[selected_date_column, selected_value_column]].dropna().copy()
                        
                        # Extract year and month
                        temp_df['year'] = temp_df[selected_date_column].dt.year
                        temp_df['month'] = temp_df[selected_date_column].dt.month
                        
                        # Get available years
                        years = sorted(temp_df['year'].unique())
                        
                        if len(years) <= 1:
                            st.warning("Not enough years in the data for year-over-year comparison.")
                        else:
                            # Create figure
                            fig = go.Figure()
                            
                            # Add line for each year
                            for year in years:
                                year_data = temp_df[temp_df['year'] == year]
                                
                                # Group by month and calculate average
                                monthly_avg = year_data.groupby('month')[selected_value_column].mean().reset_index()
                                
                                fig.add_trace(go.Scatter(
                                    x=monthly_avg['month'],
                                    y=monthly_avg[selected_value_column],
                                    mode='lines+markers',
                                    name=str(year)
                                ))
                            
                            # Update layout
                            fig.update_layout(
                                title=f"Year-over-Year Comparison of {selected_value_column}",
                                xaxis=dict(
                                    title="Month",
                                    tickmode='array',
                                    tickvals=list(range(1, 13)),
                                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                ),
                                yaxis_title=selected_value_column,
                                height=self.chart_height
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        self.error_handler.handle_error(e, "Error creating year-over-year comparison")
                        st.error("Could not create year-over-year comparison. Please check your data.")
                
                elif analysis_type == "Monthly Aggregation":
                    try:
                        # Convert to datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(df[selected_date_column]):
                            df = df.copy()
                            df[selected_date_column] = pd.to_datetime(df[selected_date_column], errors="coerce")
                        
                        # Create a copy with valid dates and values
                        temp_df = df[[selected_date_column, selected_value_column]].dropna().copy()
                        
                        # Extract year and month
                        temp_df['year_month'] = temp_df[selected_date_column].dt.strftime('%Y-%m')
                        
                        # Select aggregation method
                        agg_method = st.selectbox(
                            "Aggregation method",
                            ["Mean", "Sum", "Min", "Max", "Count"]
                        )
                        
                        # Map aggregation method to function
                        agg_func = {
                            "Mean": "mean",
                            "Sum": "sum",
                            "Min": "min",
                            "Max": "max",
                            "Count": "count"
                        }[agg_method]
                        
                        # Group by year-month and aggregate
                        monthly_data = temp_df.groupby('year_month')[selected_value_column].agg(agg_func).reset_index()
                        
                        # Sort by year-month
                        monthly_data = monthly_data.sort_values('year_month')
                        
                        # Create bar chart
                        fig = px.bar(
                            monthly_data,
                            x='year_month',
                            y=selected_value_column,
                            title=f"Monthly {agg_method} of {selected_value_column}",
                            labels={"year_month": "Month", selected_value_column: f"{agg_method} of {selected_value_column}"}
                        )
                        
                        fig.update_layout(
                            xaxis=dict(tickangle=45),
                            height=self.chart_height
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        self.error_handler.handle_error(e, "Error creating monthly aggregation")
                        st.error("Could not create monthly aggregation. Please check your data.")
            else:
                st.warning("Could not create time series plot with the selected columns.")
    
    def _render_custom_analysis(self, df: pd.DataFrame):
        """
        Render custom analysis section.
        
        Args:
            df: DataFrame to analyze
        """
        st.subheader("Custom Analysis")
        
        # Analysis type selector
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Scatter Plot Matrix", "Group Analysis", "Outlier Detection", "Custom Grouping"]
        )
        
        if analysis_type == "Scatter Plot Matrix":
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
            
            if len(numerical_columns) < 2:
                st.info("At least two numerical columns are required for scatter plot matrix.")
                return
            
            # Allow selecting columns
            selected_columns = st.multiselect(
                "Select columns for scatter plot matrix",
                numerical_columns,
                default=list(numerical_columns[:min(4, len(numerical_columns))])
            )
            
            if len(selected_columns) >= 2:
                # Create scatter plot matrix
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected_columns,
                    title="Scatter Plot Matrix",
                    height=max(600, self.chart_height)
                )
                
                fig.update_traces(diagonal_visible=False)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least two columns.")
        
        elif analysis_type == "Group Analysis":
            # Get columns
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
            
            if len(categorical_columns) == 0:
                st.info("No categorical columns found for grouping.")
                return
            
            if len(numerical_columns) == 0:
                st.info("No numerical columns found for aggregation.")
                return
            
            # Allow selecting columns
            group_column = st.selectbox("Select column to group by", categorical_columns)
            agg_column = st.selectbox("Select column to aggregate", numerical_columns)
            agg_function = st.selectbox("Select aggregation function", ["Mean", "Sum", "Count", "Min", "Max"])
            
            # Map function names to pandas function names
            agg_func_map = {
                "Mean": "mean",
                "Sum": "sum",
                "Count": "count",
                "Min": "min",
                "Max": "max"
            }
            
            # Perform grouping
            try:
                grouped = df.groupby(group_column)[agg_column].agg(agg_func_map[agg_function]).reset_index()
                
                # Sort by aggregated value in descending order
                grouped = grouped.sort_values(agg_column, ascending=False)
                
                # Limit to top N values
                top_n = st.slider("Show top N groups", 5, 50, 10)
                grouped = grouped.head(top_n)
                
                # Create chart
                fig = px.bar(
                    grouped,
                    x=group_column,
                    y=agg_column,
                    title=f"{agg_function} of {agg_column} by {group_column}",
                    height=self.chart_height
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show table
                st.write("Grouped Data:")
                st.dataframe(grouped)
            except Exception as e:
                self.error_handler.handle_error(e, "Error performing group analysis")
                st.error("Could not perform group analysis. Please check your data.")
        
        elif analysis_type == "Outlier Detection":
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
            
            if len(numerical_columns) == 0:
                st.info("No numerical columns found for outlier detection.")
                return
            
            # Allow selecting column
            column = st.selectbox("Select column for outlier detection", numerical_columns)
            
            # Select outlier detection method
            method = st.selectbox(
                "Select outlier detection method",
                ["Z-Score", "IQR (Interquartile Range)"]
            )
            
            # Detect outliers
            try:
                if method == "Z-Score":
                    # Z-Score method
                    threshold = st.slider("Z-Score threshold", 1.0, 5.0, 3.0, 0.1)
                    
                    z_scores = (df[column] - df[column].mean()) / df[column].std()
                    outliers = df[abs(z_scores) > threshold]
                    
                    st.write(f"Found {len(outliers)} outliers using Z-Score method (|z| > {threshold}).")
                
                elif method == "IQR (Interquartile Range)":
                    # IQR method
                    multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
                    
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    
                    st.write(f"Found {len(outliers)} outliers using IQR method (outside {Q1 - multiplier * IQR:.2f} to {Q3 + multiplier * IQR:.2f}).")
                
                # Create box plot
                fig = px.box(
                    df,
                    y=column,
                    title=f"Box Plot of {column} with Outliers Highlighted",
                    points="outliers",
                    height=self.chart_height
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show outliers table if any found
                if len(outliers) > 0:
                    st.write("Sample of outliers:")
                    st.dataframe(outliers.head(10))
                    
                    # Option to download outliers
                    outlier_csv = outliers.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download outliers as CSV",
                        data=outlier_csv,
                        file_name=f"outliers_{column}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                self.error_handler.handle_error(e, "Error detecting outliers")
                st.error("Could not detect outliers. Please check your data.")
        
        elif analysis_type == "Custom Grouping":
            # Get columns
            all_columns = df.columns.tolist()
            
            # Allow selecting columns
            group_columns = st.multiselect(
                "Select columns to group by",
                all_columns,
                default=all_columns[:min(2, len(all_columns))]
            )
            
            agg_column = st.selectbox(
                "Select column to aggregate",
                [col for col in all_columns if col not in group_columns]
            )
            
            agg_functions = st.multiselect(
                "Select aggregation functions",
                ["Count", "Mean", "Sum", "Min", "Max", "Std", "Median"],
                default=["Count", "Mean"]
            )
            
            # Map function names to pandas function names
            agg_func_map = {
                "Count": "count",
                "Mean": "mean",
                "Sum": "sum",
                "Min": "min",
                "Max": "max",
                "Std": "std",
                "Median": "median"
            }
            
            # Selected functions
            selected_funcs = {agg_column: [agg_func_map[func] for func in agg_functions]}
            
            if group_columns and agg_column and agg_functions:
                try:
                    # Perform grouping
                    grouped = df.groupby(group_columns).agg(selected_funcs)
                    
                    # Flatten multi-level column names
                    grouped.columns = [f"{agg_column}_{col[1]}" for col in grouped.columns]
                    
                    # Reset index
                    grouped = grouped.reset_index()
                    
                    # Show table
                    st.write("Grouped Data:")
                    st.dataframe(grouped)
                    
                    # Option to download grouped data
                    grouped_csv = grouped.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download grouped data as CSV",
                        data=grouped_csv,
                        file_name="grouped_data.csv",
                        mime="text/csv"
                    )
                    
                    # Create visualization if there are not too many groups
                    if len(grouped) <= 20 and len(group_columns) <= 2:
                        st.subheader("Visualization")
                        
                        # Select column to visualize
                        vis_column = st.selectbox(
                            "Select column to visualize",
                            grouped.columns[len(group_columns):]
                        )
                        
                        if len(group_columns) == 1:
                            # Single grouping column
                            fig = px.bar(
                                grouped,
                                x=group_columns[0],
                                y=vis_column,
                                title=f"{vis_column} by {group_columns[0]}",
                                height=self.chart_height
                            )
                        else:
                            # Two grouping columns
                            fig = px.bar(
                                grouped,
                                x=group_columns[0],
                                y=vis_column,
                                color=group_columns[1],
                                barmode="group",
                                title=f"{vis_column} by {group_columns[0]} and {group_columns[1]}",
                                height=self.chart_height
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    self.error_handler.handle_error(e, "Error performing custom grouping")
                    st.error("Could not perform custom grouping. Please check your data and selected columns.")