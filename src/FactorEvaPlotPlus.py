import datetime
import json5
import pandas as pd
import dolphindb as ddb
import numpy as np
from typing import Dict
from functools import lru_cache
import streamlit as st
from FactorEva import SingleFactorBackTest, OptimizeFactorBackTest, OptimizeFactorAttribute
import plotly.express as px
import plotly.graph_objects as go

COLORS = ["#5292C5", "#DEA3BC", "#985C8C", "#E1646E", "#ffa26f"]
# st.set_page_config(
#     page_title="Factor Evaluation",
#     # layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # 适中的内边距设置
# st.markdown("""
# <style>
#     .main .block-container {
#         max-width: 1200px;
#         padding-top: 2rem;
#         padding-right: 3rem;
#         padding-left: 3rem;
#         padding-bottom: 2rem;
#     }
#
#     /* 增加元素间距 */
#     .stVerticalBlock {
#         gap: 1.5rem;
#     }
#
#     /* 图表容器适中间距 */
#     .element-container {
#         margin-bottom: 1.5rem;
#     }
# </style>
# """, unsafe_allow_html=True)


def plotlyPlot(data: pd.DataFrame, xCol: str, yCol: list, figType: str, title: str = None) -> go.Figure:
    """plotly Figure with 10*6 size

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    xCol : str
        x轴列名
    yCol : list
        y轴列名列表
    figType : str
        图表类型：'line' 或 'bar'
    title : str, optional
        图表标题

    Returns:
    --------
    go.Figure
        plotly图形对象
    """

    # 定义颜色方案
    colors = COLORS

    # 创建基础图形
    fig = go.Figure()

    if figType.lower() == 'line':
        # 线状图
        for i, column in enumerate(yCol):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=data[xCol],
                y=data[column],
                mode='lines',
                name=column,
                line=dict(color=color, width=2.5),
                opacity=0.9,
                hovertemplate=f'<b>{column}</b><br>{xCol}: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
            ))

    elif figType.lower() == 'bar':
        # 柱状图 - 设置更窄的柱子
        for i, column in enumerate(yCol):
            color = colors[i % len(colors)]
            fig.add_trace(go.Bar(
                x=data[xCol],
                y=data[column],
                name=column,
                marker_color=color,
                opacity=0.9,
                width=0.4,  # 设置柱子宽度（默认约为0.8）
                hovertemplate=f'<b>{column}</b><br>{xCol}: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
            ))

        fig.update_layout(
            barmode='group',
            bargap=0.3,  # 柱子组之间的间距
            bargroupgap=0.1  # 同一组内柱子之间的间距
        )

    else:
        raise ValueError("figType must be 'line' or 'bar'")

    # 设置10*6尺寸（1000x600像素）
    fig.update_layout(
        width=1200,  # 10英寸 ≈ 1000像素
        height=600,  # 6英寸 ≈ 600像素
        template="plotly_white",
        font=dict(size=12),
        margin=dict(l=60, r=50, t=80, b=60),  # 增加上边距为图例留空间
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,  # 保持在图表上方
            xanchor="center",  # 水平居中对齐
            x=0.5,  # 水平位置居中
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.9)',
            # 去掉边框
            bordercolor='rgba(255,255,255,0)',  # 透明边框
            borderwidth=0  # 边框宽度设为0
        )
    )

    # 添加标题
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16),
                y=0.95  # 调整标题位置，为图例留出空间
            )
        )

    # 更新坐标轴标签 - 关键修改：调整刻度标签位置
    fig.update_xaxes(
        title_text="",  # 空字符串，去掉X轴标题
        tickangle=30,
        ticklabelposition="outside right",  # 让标签更靠近轴线
        # 调整标签与轴线的距离（负值表示更靠近）
        # ticklabelstandoff=0,
        # 调整刻度与轴线的距离
        ticks="outside",
        ticklen=1,
        # 如果需要进一步调整，可以设置边距
        # automargin=False
    )

    fig.update_yaxes(
        title_text="Value",
        title_standoff=10
    )

    # 如果是时间序列数据，优化x轴显示
    if pd.api.types.is_datetime64_any_dtype(data[xCol]):
        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickangle=0,
            ticklabelposition="outside right",  # 时间序列也应用相同设置
            # ticklabelstandoff=0,
            ticks="outside",
            ticklen=1
        )

    return fig


class SingleFactorPlot(SingleFactorBackTest):
    def __init__(self, session: ddb.session, pool: ddb.DBConnectionPool, config: Dict):
        # 基本信息
        super().__init__(session, pool, config)

    @lru_cache(128)
    def get_SummaryData(self, r_interval: int) -> Dict:
        resDict = self.session.run(f"""
                pt = select ReturnInterval,period,factor,indicator,value from loadTable("{self.resultDB}","{self.summary_table}") 
                    where ReturnInterval == int({r_interval}) and indicator in ["IC","RankIC"]
                template_pt = select startDate as date,period from loadTable("{self.combineDB}","{self.template_table}")
                // 添加时间
                pt = lj(template_pt,pt,`period);
                update pt set yearInt = year(date);
                update pt set yearStr = "Year"+string(yearInt)
                year_list = sort(exec distinct(yearInt) from pt)
                undef(`template_pt);

                // avg(IC)
                TotalIC_pt = select avg(value) as Total from pt where indicator == "IC" group by factor
                sortBy!(TotalIC_pt,`factor)
                YearIC_pt = select avg(value) as value from pt where indicator == "IC" pivot by factor, yearStr
                YearIC_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearIC_pt).eval()
                TotalIC_pt = lj(TotalIC_pt, YearIC_pt, `factor)

                // avg(RankIC)
                TotalRankIC_pt = select avg(value) as Total from pt where indicator == "RankIC" group by factor 
                sortBy!(TotalRankIC_pt,`factor)
                YearRankIC_pt = select avg(value) as value from pt where indicator == "RankIC" pivot by factor, yearStr
                YearRankIC_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearRankIC_pt).eval()
                TotalRankIC_pt = lj(TotalRankIC_pt, YearRankIC_pt, `factor)

                // avg(IC)\std(IC)
                TotalICIR_pt = select avg(value)\std(value) as Total from pt where indicator == "IC" group by factor
                sortBy!(TotalICIR_pt,`factor)
                YearICIR_pt = select avg(value)\std(value) as value from pt where indicator == "IC" pivot by factor, yearStr
                YearICIR_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearICIR_pt).eval()
                TotalICIR_pt = lj(TotalICIR_pt, YearICIR_pt, `factor)

                // avg(RankIC)\std(RankIC)
                TotalRankICIR_pt = select avg(value)\std(value) as Total from pt where indicator == "RankIC" group by factor
                sortBy!(TotalRankICIR_pt,`factor)
                YearRankICIR_pt = select avg(value)\std(value) as value from pt where indicator == "RankIC" pivot by factor, yearStr
                YearRankICIR_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearRankICIR_pt).eval()
                TotalRankICIR_pt = lj(TotalRankICIR_pt, YearRankICIR_pt, `factor)

                // 返回结果
                res_dict = dict(["TotalIC","TotalRankIC","TotalICIR","TotalRankICIR"], [TotalIC_pt,TotalRankIC_pt,TotalICIR_pt,TotalRankICIR_pt])
                res_dict
                """)
        return resDict

    def Summary_plot(self):
        """
        所有因子横向比较可视化
        including: avg(IC), avg(RankIC), ICIR
        """
        r_interval = st.selectbox(
            label="请输入未来收益率区间长度",
            options=(i for i in self.returnIntervals),
            index=0,
            format_func=str,
            help='即ReturnModel中的returnIntervals'
        )
        st.title("_Total Factor Performance Comparison_")
        Dict = self.get_SummaryData(r_interval=r_interval)
        TotalIC_df = Dict["TotalIC"]
        TotalRankIC_df = Dict["TotalRankIC"]
        TotalICIR_df = Dict["TotalICIR"]
        TotalRankICIR_df = Dict["TotalRankICIR"]
        st.subheader("All Factors' avg(IC)", divider=True)
        st.dataframe(data=TotalIC_df)
        st.subheader("All Factors' avg(RankIC)", divider=True)
        st.dataframe(data=TotalRankIC_df)
        st.subheader("All Factors' ICIR", divider=True)
        st.dataframe(data=TotalICIR_df)
        st.subheader("All Factors' RankICIR", divider=True)
        st.dataframe(data=TotalRankICIR_df)

    @lru_cache(128)
    def get_factorData(self, factor: str, returnInterval: int) -> Dict:
        resDict = self.session.run(rf"""
        pt=select * from loadTable("{self.resultDB}","{self.summary_table}") 
            where factor == "{factor}" and ReturnInterval=={returnInterval}
        quantile_pt=select * from loadTable("{self.resultDB}","{self.quantile_table}") 
                    where factor == "{factor}";

        // 因子收益率&累计因子收益率
        R=select value from pt where indicator ="R_OLS" pivot by TradeTime,indicator;
        R_cumsum=R.copy();
        L=R_cumsum["TradeTime"];
        dropColumns!(R_cumsum,`TradeTime);
        R_cumsum=cumsum(R_cumsum);
        R_cumsum=select L as TradeTime,* from R_cumsum;

        // Reg_stat
        Obs=select value from pt where indicator == "Obs" pivot by TradeTime,indicator;
        Std_Error=select value from pt where indicator == "Std_Error" pivot by TradeTime,indicator;  // 残差标准差
        R_square=select value from pt where indicator == "R_square" pivot by TradeTime,indicator;
        Adj_square=select value from pt where indicator == "Adj_square" pivot by TradeTime,indicator;

        // Tstat
        t_stat = select value from pt where indicator == "R_tstat" pivot by TradeTime,indicator;
        // alpha_tStat = select value from pt where indicator == "Alpha_tstat" pivot by TradeTime,indicator;

        // IC & 累计IC
        IC=select value from pt where indicator="IC" pivot by TradeTime,indicator;
        IC_cumsum=IC.copy();
        L=IC_cumsum["TradeTime"];
        dropColumns!(IC_cumsum,`TradeTime);
        IC_cumsum=cumsum(IC_cumsum);
        IC_cumsum=select L as TradeTime,* from IC_cumsum;

        // RankIC & 累计RankIC
        RankIC=select value from pt where indicator="RankIC" pivot by TradeTime,indicator;
        RankIC_cumsum=RankIC.copy();
        L=RankIC_cumsum["TradeTime"];
        dropColumns!(RankIC_cumsum,`TradeTime);
        RankIC_cumsum=cumsum(RankIC_cumsum);
        RankIC_cumsum=select L as TradeTime,* from RankIC_cumsum;

        // Yearly avg(IC)&IR
        data=unpivot(IC,keyColNames="TradeTime",valueColNames=columnNames(IC)[1:])
        rename!(data,`TradeTime`factor`factor_IC);
        avg_IC=select avg(factor_IC) from data pivot by year(TradeTime) as year,factor;
        IR=select avg(factor_IC)/std(factor_IC) from data pivot by year(TradeTime) as year,factor;

        // Yearly avg(RankIC)&RankIR
        data=unpivot(RankIC,keyColNames="TradeTime",valueColNames=columnNames(RankIC)[1:])
        rename!(data,`TradeTime`factor`factor_RankIC);
        avg_RankIC=select avg(factor_RankIC) from data pivot by year(TradeTime) as year,factor;
        RankIR=select avg(factor_RankIC)/std(factor_RankIC) from data pivot by year(TradeTime) as year,factor;

        // 返回为字典格式
        Dict=dict(["R_square","Adj_square","Obs","Std_Error","R","R_cumsum","t_stat",
                  "IC","IC_cumsum","RankIC","RankIC_cumsum","avg_IC","IR","avg_RankIC","RankIR"],
                  [R_square,Adj_square,Obs,Std_Error,R,R_cumsum,t_stat,IC,IC_cumsum,RankIC,RankIC_cumsum,avg_IC,IR,avg_RankIC,RankIR]);

        // Quantile Return & Quantile Cumsum Return
        returnIntervals = {self.returnIntervals}
        for (r_interval in returnIntervals){{  // 这里只统计累计值(cumsum)
            df = sql(select=[sqlCol(`TradeTime)].append!(sqlCol("QuantileReturn"+string(1..{self.Quantile}))),
                    from=quantile_pt, where=<ReturnInterval == r_interval>).eval()
            ts_list = df[`TradeTime];
            dropColumns!(df,`TradeTime);
            df = cumsum(df) + 1
            Dict["Return"+string(r_interval)] = select ts_list as `TradeTime, * from df
        }}
        undef(`pt); // 清除缓存

        Dict
        """)
        return resDict

    def FactorR_plot(self):
        """单因子收益率可视化
        including: R/IC/RankIC/Tstats/Reg_stats(R_square/Adj_square/Obs)
        """
        r_interval = st.selectbox(
            label="请输入未来收益率区间长度",
            options=(i for i in self.returnIntervals),
            index=0,
            format_func=str,
            help='即ReturnModel中的returnIntervals'
        )
        factor = st.selectbox(
            label="请选择因子",
            options=self.factor_list,
            index=0,
            format_func=str,
            help="选择当前因子进行因子分层收益展示"
        )
        st.title("_Single Factor BackTest Analysis_")
        tabReg, tabIC, tabQuantile, tabStats = st.tabs(["回归法", "IC法", "分层回测", "其他指标"])
        Dict = self.get_factorData(factor=factor, returnInterval=r_interval)
        R_square = Dict["R_square"]
        Adj_square = Dict["Adj_square"]
        Obs = Dict["Obs"]
        Std_Error = Dict["Std_Error"]
        R = Dict["R"]
        R_cumsum = Dict["R_cumsum"]
        t_stat = Dict["t_stat"]
        # IC=Dict["IC"]
        IC_cumsum = Dict["IC_cumsum"]
        # RankIC=Dict["RankIC"]
        RankIC_cumsum = Dict["RankIC_cumsum"]
        avg_IC = Dict["avg_IC"]
        IR = Dict["IR"]
        avg_RankIC = Dict["avg_RankIC"]
        RankIR = Dict["RankIR"]
        with tabReg:
            st.subheader("Single Factor Return", divider=True)
            fig = plotlyPlot(R, "TradeTime", R.columns.tolist()[1:], "line")
            st.plotly_chart(fig, use_container_width=True)
            # st.line_chart(data=R, x="TradeTime", y=None)
            st.subheader("Single Factor Return(cumsum)", divider=True)
            fig = plotlyPlot(R_cumsum, "TradeTime", R_cumsum.columns.tolist()[1:], "line")
            st.plotly_chart(fig, use_container_width=True)
            # st.line_chart(data=R_cumsum, x="TradeTime", y=None)

            st.subheader("Factor Tstat", divider=True)
            fig = plotlyPlot(t_stat, "TradeTime", t_stat.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=t_stat, x="TradeTime", y=None, stack=False)
            st.write("T值绝对值大于等于2的比例")
            t_stat = t_stat.set_index("TradeTime")
            t_stat = (t_stat.abs() >= 2).mean()  # .mean()计算|T|≥2的比例
            st.dataframe(data=t_stat)
        with tabIC:
            # st.subheader("Factor IC",divider=True)
            # st.bar_chart(data=IC,x="TradeTime",y=None,stack=False)
            # st.subheader("Factor RankIC",divider=True)
            # st.bar_chart(data=RankIC,x="TradeTime",y=None,stack=False)

            st.subheader("Factor IC(cumsum)", divider=True)
            fig = plotlyPlot(IC_cumsum, "TradeTime", IC_cumsum.columns.tolist()[1:], "line")
            st.plotly_chart(fig, use_container_width=True)
            # st.line_chart(data=IC_cumsum, x="TradeTime", y=None)

            st.subheader("Factor RankIC(cumsum)", divider=True)
            fig = plotlyPlot(RankIC_cumsum, "TradeTime", RankIC_cumsum.columns.tolist()[1:], "line")
            st.plotly_chart(fig, use_container_width=True)
            # st.line_chart(data=RankIC_cumsum, x="TradeTime", y=None)

            st.subheader("Factor avg(IC)", divider=True)
            fig = plotlyPlot(avg_IC, "year", avg_IC.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=avg_IC, x="year", y=None, stack=False)
            st.dataframe(data=avg_IC)
            st.write("Total avg(IC):")
            st.dataframe(data=avg_IC.set_index("year").mean())

            st.subheader("Factor IR", divider=True)
            fig = plotlyPlot(IR, "year", IR.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=IR, x="year", y=None, stack=False)
            st.dataframe(data=IR)

            st.subheader("Factor avg(RankIC)", divider=True)
            fig = plotlyPlot(avg_RankIC, "year", avg_RankIC.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=avg_RankIC, x="year", y=None, stack=False)
            st.dataframe(data=avg_RankIC)
            st.write("Total avg(RankIC):")
            st.dataframe(data=avg_RankIC.set_index("year").mean())

            st.subheader("Factor RankIR", divider=True)
            fig = plotlyPlot(RankIR, "year", RankIR.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=RankIR, x="year", y=None, stack=False)
            st.dataframe(data=RankIR)
        with tabQuantile:
            for r_interval in self.returnIntervals:
                st.subheader(f"Single Factor Quantile Return(ReturnInterval={r_interval})", divider=True)
                data = Dict["Return" + str(r_interval)]
                fig = plotlyPlot(data, "TradeTime", data.columns.tolist()[1:], "line")
                st.plotly_chart(fig, use_container_width=True)
                # st.line_chart(data=Dict["Return" + str(r_interval)], x="TradeTime", y=None)
        with tabReg:
            st.subheader("R square", divider=True)
            fig = plotlyPlot(R_square, "TradeTime", R_square.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=R_square, x="TradeTime", y=None, stack=False)

            st.subheader("Adj R suqare", divider=True)
            fig = plotlyPlot(Adj_square, "TradeTime", Adj_square.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=Adj_square, x="TradeTime", y=None, stack=False)

            st.subheader("Std Error(残差标准差)", divider=True)
            fig = plotlyPlot(Std_Error, "TradeTime", Std_Error.columns.tolist()[1:], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=Std_Error, x="TradeTime", y=None, stack=False)

            st.subheader("Num of Obs", divider=True)
            fig = plotlyPlot(Obs, "TradeTime", Obs.columns.tolist()[1:], "line")
            st.plotly_chart(fig, use_container_width=True)
            # st.line_chart(data=Obs, x="TradeTime", y=None)

        return Dict  # 返回绘图用的数据Dictionary


class OptimizeFactorPlot(OptimizeFactorBackTest):
    def __init__(self, session: ddb.session, pool: ddb.DBConnectionPool, config: Dict, ):
        super(OptimizeFactorPlot, self).__init__(session, pool, config, Optimize_callBackFunc=None)
        self.barReturnCol = config["barReturnLabel"]
        self.weightName = self.get_weightName()

    @lru_cache(128)
    def get_weightName(self):
        """获取所有数据库中的权重标签"""
        t = self.session.run(f"""
        select count(*) from loadTable("{self.weightDB}","{self.weightTB}") group by weightName
        """)
        return sorted(t["weightName"])

    @lru_cache(128)
    def Optimize_Plot(self):
        weightName = st.selectbox(
            label="请输入权重方法名称",
            options=(i for i in self.weightName),
            index=0,
            format_func=str,
            help='即weight database中的weightName标签'
        )
        tabOpt, tabOther = st.tabs(["组合优化回测", "Other"])

        # 进行可视化
        # st.title("_Optimize Factor BackTest_")
        with tabOpt:
            st.subheader("Optimize Return", divider=True)
            data = self.session.run(f"""
                    // 权重数据
                    weight_df = select symbol,concatDateTime(TradeDate,TradeTime) as TradeTime,weight 
                               from loadTable("{self.weightDB}","{self.weightTB}")
                               where weightName == "{weightName}" and (TradeDate between {self.start_dot_date} and {self.end_dot_date})

                    // 行情数据
                    label_df = select symbol,concatDateTime(TradeDate,TradeTime) as TradeTime,label 
                                from loadTable("{self.labelDB}","{self.labelTB}")
                                where labelName == "{self.barReturnCol}" and (TradeDate between {self.start_dot_date} and {self.end_dot_date})

                    label_df = lsj(label_df,weight_df,`symbol`TradeTime);
                    update label_df set weight = nullFill(weight,0.0);
                    update label_df set weight = weight/sum(weight) context by TradeTime

                    result_df = select label**weight as Return from label_df group by TradeTime
                    sortBy!(result_df,`TradeTime) 
                    update result_df set CumReturn = 1.0+cumsum(Return)
                    result_df
                    """)
            # st.bar_chart(data=data, x="TradeTime", y="Return", stack=False)
            fig = plotlyPlot(data, "TradeTime", ["Return"], "bar")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Optimize Return(cumsum)", divider=True)
            fig = plotlyPlot(data, "TradeTime", ["CumReturn"], "line")
            st.plotly_chart(fig, use_container_width=True)
            # st.line_chart(data=data, x="TradeTime", y="CumReturn")


class OptimizeFactorAttrPlot(OptimizeFactorAttribute):
    def __init__(self, session: ddb.session, pool: ddb.DBConnectionPool, config: Dict):
        super(OptimizeFactorAttrPlot, self).__init__(session, pool, config, Data_prepareFunc=None)

    def Attribute_Plot(self):
        ts = pd.Timestamp(self.start_date)
        d = st.date_input("Please Choose the Attr Date", datetime.date(ts.year, ts.month, ts.day))
        # tabPure, tabReturn, tabRisk = st.tabs(["纯因子收益率","主动收益归因","主动风险归因"])
        tabAttr, tabOther = st.tabs(["因子归因", "Other"])
        strDate = pd.Timestamp(d).strftime("%Y.%m.%d")

        # # 进行可视化
        # st.title("_Optimize Factor Attribution_")
        with tabAttr:
            st.subheader("Pure Factor Return", divider=True)
            pureFactorRetData = self.session.run(f"""
            df = select * from loadTable("{self.resultDB}","{self.pureFactorRet_table}") 
                where TradeDate == {strDate};
            df = unpivot(df, keyColNames=`TradeDate, valueColNames=columnNames(df)[1:])
            rename!(df,`TradeDate`indicator`value);
            df
            """)
            fig = plotlyPlot(pureFactorRetData, "indicator", ["value"], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=pureFactorRetData, x="indicator", y="value", stack=False)
            # with tabReturn:
            st.subheader("Active Return Attribution", divider=True)
            ActiveReturnData = self.session.run(f"""
            df = select * from loadTable("{self.resultDB}","{self.activeRetAttr_table}") where TradeDate == {strDate}
            df = unpivot(df, keyColNames=`TradeDate, valueColNames=columnNames(df)[1:])
            rename!(df,`TradeDate`indicator`value);
            df
            """)
            # st.bar_chart(data=ActiveReturnData, x="indicator", y="value", stack=False)
            fig = plotlyPlot(ActiveReturnData, "indicator", ["value"], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # with tabRisk:
            st.subheader("Active Risk Attribution", divider=True)
            ActiveRiskData = self.session.run(f"""
            df = select * from loadTable("{self.resultDB}","{self.activeRiskAttr_table}") where TradeDate == {strDate}
            df = unpivot(df, keyColNames=`TradeDate, valueColNames=columnNames(df)[1:])
            rename!(df,`TradeDate`indicator`value);
            df
            """)
            fig = plotlyPlot(ActiveRiskData, "indicator", ["value"], "bar")
            st.plotly_chart(fig, use_container_width=True)
            # st.bar_chart(data=ActiveRiskData, x="indicator", y="value", stack=False)


if __name__ == "__main__":
    session = ddb.session()
    session.connect("172.16.0.184", 8001, "maxim", "dyJmoc-tiznem-1figgu")
    pool = ddb.DBConnectionPool("172.16.0.184", 8001, 10, "maxim", "dyJmoc-tiznem-1figgu")
    with open(r"D:\DolphinDB\Project\FactorEva\src\config\factorEva_cfg.json5", mode="r",
              encoding="UTF-8") as file:
        cfg = json5.load(file)
    P = SingleFactorPlot(session=session, pool=pool, config=cfg)
    # P.Summary_plot()
    P.FactorR_plot()  # 绘制单因子模型回测结果

    # with open(r"D:\DolphinDB\Project\FactorEva\src\config\factorOpt_cfg.json5", mode="r",
    #           encoding="UTF-8") as file:
    #     cfg = json5.load(file)
    # W = OptimizeFactorPlot(session=session, pool=pool, config=cfg)
    # W.Optimize_Plot()

    # with open(r"D:\DolphinDB\Project\FactorEva\src\config\factorAttr_cfg.json5", mode="r",
    #           encoding="UTF-8") as file:
    #     cfg = json5.load(file)
    # A = OptimizeFactorAttrPlot(session=session, pool=pool, config=cfg)
    # A.Attribute_Plot()

