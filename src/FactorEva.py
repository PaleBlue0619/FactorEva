import os,sys
import json,json5
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import *
pd.set_option("display.max_columns",None)
plt.rcParams['font.sans-serif']=['KaiTi'] # 显示中文
plt.rcParams['axes.unicode_minus']=False # 显示负号

class SingleFactorBackTest:
    def __init__(self, session, pool, config,
        Label_prepareFunc=None,
        Combine_prepareFunc=None
    ):
        # 基本信息
        self.strategy_name="strategy"
        self.session=session
        self.pool=pool

        # 库表类
        self.labelDB = config["label_database"]
        self.labelTB = config["label_table"]
        self.factorDB = config["factor_database"]
        self.factorTB = config["factor_table"]
        self.combineDB= config["combine_database"]
        self.combineTB= config["combine_table"]
        self.resultDB= config["result_database"]
        self.quantile_table= config["quantile_table"]      # 分层回测结果表
        self.summary_table= config["summary_table"]        # ICIR&回归法结果表

        # 函数类
        self.Label_prepareFunc=Label_prepareFunc     # 标签准备函数
        self.Combine_prepareFunc=Combine_prepareFunc # 数据合并函数

        # 变量类
        self.start_date=config["start_date"]
        self.end_date=config["end_date"]
        self.start_dot_date=pd.Timestamp(self.start_date).strftime('%Y.%m.%d')
        self.end_dot_date=pd.Timestamp(self.end_date).strftime('%Y.%m.%d')
        self.factor_list=config["factor_list"]
        self.dailyFreq = config["dailyFreq"]    # 是否为日频
        self.callBackPeriod=int(config["callBackPeriod"]) # 回看周期
        self.returnIntervals = config["returnIntervals"] # [1,5,10,20,30] 区间收益率长度
        self.Quantile=int(config["quantile"])  # 因子分组收益率统计
        self.dailyPnlLimit = config["dailyPnlLimit"]
        self.useMinFreqPeriod = config["useMinFreqPeriod"]
        self.idCol = config["idCol"]
        self.barReturnCol = config["barReturnLabel"]
        self.futureReturnCols = config["futureReturnLabels"]

        # 中间计算+最终结果类
        self.template_table="template"
        self.template_individual_table="template_individual"

    def init_labelDB(self, dropDB: bool = False, dropTB: bool = False):
        """
        初始化标签数据库
        """
        if dropDB and self.session.existsDatabase(self.labelDB):
            self.session.dropDatabase(self.labelDB)
        if dropTB and self.session.existsTable(dbUrl=self.labelDB, tableName=self.labelTB):
            self.session.dropTable(dbPath=self.labelDB, tableName=self.labelTB)
        if not self.session.existsTable(self.labelDB, self.labelTB):
            """新建数据库表"""
            self.session.run(f"""
                db = database("{self.labelDB}",VALUE,2010.01M..2030.01M, engine="TSDB")
                schemaTb = table(1:0, ["symbol","TradeDate","TradeTime","labelName","label","MaxDate","MaxTime"],
                                    [SYMBOL,DATE,TIME,STRING,DOUBLE,DATE,TIME])
                db.createPartitionedTable(schemaTb, "{self.labelTB}", partitionColumns=`TradeDate, 
                    sortColumns=`labelName`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
            """)

    def init_combineDB(self, dropDB=True):
        """[Necessary]初始化合并数据库+模板数据库"""
        if dropDB:
            if self.session.existsDatabase(dbUrl=self.combineDB):  # 删除数据库
                self.session.dropDatabase(dbPath=self.combineDB)
        # Combine Table 默认每次回测前删除上次的因子库（因为因子个数名称&调仓周期可能不一样）
        if self.session.existsTable(dbUrl=self.combineDB,tableName=self.combineTB):
            self.session.dropTable(dbPath=self.combineDB,tableName=self.combineTB)
        columns_name=["symbol","TradeDate","TradeTime"]+[self.barReturnCol]+self.futureReturnCols+self.factor_list+["period"]
        columns_type=["SYMBOL","DATE","TIME","DOUBLE"]+["DOUBLE"]*len(self.returnIntervals)+["DOUBLE"]*len(self.factor_list)+["INT"]
        self.session.run(f"""
        db=database("{self.combineDB}",VALUE,2010.01M+(0..30)*12,engine="TSDB");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.combineTB}",
            partitionColumns="TradeDate",sortColumns=["symbol","TradeDate","TradeTime"],keepDuplicates=LAST)
        """)

        # Template Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combineDB,tableName=self.template_table):
            self.session.dropTable(dbPath=self.combineDB,tableName=self.template_table)
        columns_name=["period","startDate","startTime","endDate","endTime"]
        columns_type=["INT","DATE","TIME","DATE","TIME"]
        self.session.run(f"""
        db=database("{self.combineDB}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_table}",
            partitionColumns="startDate",sortColumns=["startDate"])
        """)
        if self.session.existsTable(dbUrl=self.combineDB,tableName=self.template_individual_table):
            self.session.dropTable(dbPath=self.combineDB,tableName=self.template_individual_table)
        columns_name=["symbol","period","TradeDate","TradeTime"]
        columns_type=["SYMBOL","INT","DATE","TIME"]
        self.session.run(f"""
        db=database("{self.combineDB}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_individual_table}",
            partitionColumns="TradeDate",sortColumns=["symbol","TradeDate","TradeTime"],keepDuplicates=LAST)
        """)

    def init_ResultDataBase(self,dropTB=False):
        """单因子&多因子结果&Structured Data数据库"""
        if dropTB:
            if self.session.existsDatabase(dbUrl=self.resultDB):  # 删除数据库
                self.session.dropDatabase(dbPath=self.resultDB)
            else:
                pass
        # 单因子模型数据库
        self.session.run(f"""
        db=database("{self.resultDB}",VALUE,2020.01M..2030.01M,engine="OLAP");
        """)

        # quantile_result(avg(return) group by quantile)
        if self.session.existsTable(dbUrl=self.resultDB,tableName=self.quantile_table):
            self.session.dropTable(dbPath=self.resultDB,tableName=self.quantile_table)
        columns_name=["factor","ReturnInterval","period"]+["QuantileReturn"+str(i) for i in range(1,self.Quantile+1)]+["TradeTime"]
        columns_type=["SYMBOL","INT","INT"]+["DOUBLE"]*self.Quantile+["TIMESTAMP"]  # 分钟频因子回测
        self.session.run(f"""
            db=database("{self.resultDB}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createDimensionTable(table=schemaTb,tableName="{self.quantile_table}")
        """)

        # summary_result(reg eval+Factor Return+Factor IC+Factor t)
        if self.session.existsTable(dbUrl=self.resultDB,tableName=self.summary_table):
            self.session.dropTable(dbPath=self.resultDB,tableName=self.summary_table)
        columns_name=["factor","ReturnInterval","period","indicator","value","TradeTime"]
        columns_type=["SYMBOL","INT","INT","SYMBOL","DOUBLE","TIMESTAMP"]
        self.session.run(f"""
            db=database("{self.resultDB}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createDimensionTable(table=schemaTb,tableName="{self.summary_table}")
        """)

    def add_LabelData(self):
        if isinstance(self.Label_prepareFunc, list):    # 如果有多个标签函数
            for labelFunc in self.Label_prepareFunc:
                labelFunc(self)
        else:   # 如果只传入一个标签函数
            self.Label_prepareFunc(self)

    def add_CombineData(self):
        self.Combine_prepareFunc(self)

    def init_def(self):
        return rf"""
        def InsertData(DBName, TBName, data, batchsize){{
            // 预防Out of Memory，分批插入数据，batchsize为每次数据的记录数
            start_idx = 0
            end_idx = batchsize
            krow = rows(data)
            do{{ 
                slice_data = data[start_idx:min(end_idx,krow),]
                if (rows(slice_data)>0){{
                loadTable(DBName, TBName).append!(slice_data);
                print(start_idx);
                }}
                start_idx = start_idx + batchsize
                end_idx = end_idx + batchsize
            }}while(start_idx < krow)
        }};
    
        def RegStats(df, factor_list, ReturnInterval, callBackPeriod, currentPeriod){{
            /* ICIR & 回归法统计函数 */
            // 统计函数(peach并行内部)
            if (callBackPeriod != 1){{
                data = select * from df where currentPeriod-callBackPeriod < period <= currentPeriod;
            }}else{{
                data = select * from df where period = currentPeriod;
            }};
            rename!(data,`Ret+string(ReturnInterval), `Ret);
            counter = 0;
            for (factorName in factor_list){{
                data[`factor] = double(data[factorName])  // 为了避免数据类型发生变化
                // 有可能因子存在大量空值/Inf/返回一样的值, 使得回归统计量报错
                if (countNanInf(double(data[`factor]),true)<size(data[`factor])*0.9 and all(data[`factor]==data[`factor][0])==0){{
                    // OLS回归统计量(因子收益率/T值/R方/调整后的R方/StdError/样本数量)
                    result_OLS=ols(data[`Ret], data[`factor], intercept=true, mode=2); 
                    if (not isVoid(result_OLS[`RegressionStat])){{
                        beta_df = select factor as indicator, beta as value from result_OLS[`Coefficient];
                        beta_df[`indicator] = ["Alpha_OLS","R_OLS"] // 截距项/回归系数
                        tstat_df= select factor as indicator, tstat as value from result_OLS[`Coefficient];
                        tstat_df[`indicator] = ["Alpha_tstat", "R_tstat"] // 截距项T值/回归系数T值
                        RegDict=dict(result_OLS[`RegressionStat][`item], result_OLS[`RegressionStat][`statistics]); 
                        R_square=RegDict[`R2];
                        Adj_square=RegDict[`AdjustedR2];
                        Std_error=RegDict[`StdError];
                        Obs=RegDict['Observations'];

                        // IC统计量
                        IC_df = select `IC as indicator, corr(zscore(factor), Ret) as value from data
                        RankIC_df = select `RankIC as indicator, corr(rank(factor), rank(Ret)) as value from data
                
                        // 合并结果
                        summary_result=table([`R_square,`Adj_square,`Std_Error,`Obs] as `indicator, 
                                [R_square, Adj_square, Std_error, Obs] as `value);
                        summary_result.append!(beta_df)
                        summary_result.append!(tstat_df)
                        summary_result.append!(IC_df)
                        summary_result.append!(RankIC_df)
                        if (counter == 0){{
                            res = select factorName as factor, ReturnInterval as returnInterval, 
                                    currentPeriod as period, indicator, value from summary_result;
                        }}else{{
                            res.append!(select factorName as factor, ReturnInterval as returnInterval, 
                                    currentPeriod as period, indicator, value from summary_result);
                        }};
                        counter += 1
                    }};
                }}:
                dropColumns!(data,`factor)
            }};
            if (counter>0){{
                return res 
            }}
        }}

        def QuantileStats(df, idCol, factor_list, ReturnInterval, quantiles, currentPeriod){{
            // 分层统计函数
            // 统计函数(peach并行内部)
            data = select * from df where period == currentPeriod;
            // 按照这一个时刻数据(quantile_df)的因子值进行分组
            quantile_df = select * from df where quantilePeriod == currentPeriod-currentPeriod%ReturnInterval
            quantile_df[`id] = quantile_df[idCol]
            quantile_df = select * from quantile_df context by id limit 1; // 取第一个因子值进行分组
            bins = (1..(quantiles-1)*(1.0\quantiles)); // quantile bins
    
            counter = 0
            for (factorName in factor_list){{
                // 分层测试
                quantileFunc = quantile{{quantile_df[factorName],,"midpoint"}}; // 函数部分化应用
                split = each(quantileFunc, bins); // 按照阈值得到分割点
                quantile_df[`Quantile] = 1+digitize(quantile_df[factorName], split, right=true);
                quantile_dict = dict(quantile_df[`id], quantile_df[`Quantile])  // 当前因子的分组情况
                data[`Quantile] = quantile_dict[data[idCol]]
                quantile_return = select factorName as factor, nullFill(avg(period_return),0.0) as value from data group by Quantile
                tab = select value from quantile_return pivot by factor, Quantile
                rename!(tab, [`factor].append!(`QuantileReturn+string(columnNames(tab)[1:])))
                quantile_list = `QuantileReturn+string(1..quantiles)
                for (col in quantile_list){{
                    if (not (col in columnNames(tab))){{
                        tab[col] = 0.0; // 说明没有当前分组的数据
                    }};
                }};        
                // 合并结果
                QuantileReturn_df = sql(select=[sqlCol(`factor)].append!(sqlCol(quantile_list)), from=tab).eval()
                if (counter == 0){{        
                    qes = sql(select=[sqlCol(`factor), sqlColAlias(<ReturnInterval>, `returnInterval), sqlColAlias(<currentPeriod>, `period)].append!(sqlCol(quantile_list)), from=QuantileReturn_df).eval()           
                }}else{{
                    qes.append!(sql(select=[sqlCol(`factor), sqlColAlias(<ReturnInterval>, `returnInterval), sqlColAlias(<currentPeriod>, `period)].append!(sqlCol(quantile_list)), from=QuantileReturn_df).eval())     
                }};
                counter += 1
            }};
            if (counter>0){{
                return qes // 返回分层回测结果 
            }}
        }}

        def SingleFactorAnalysis(df, factor_list, idCol, timeCol, barReturnCol, futureReturnCols, returnIntervals, dailyFreq, callBackPeriod=1, quantiles=5, 
            dailyPnlLimit=NULL, useMinFreqPeriod=true){{
            /*单因子测试, 输出一张窄表
            totalData: GPLearnProcessing输出的因子结果+行情数据
            factor_list: 单因子列表
            idCol: 标的列
            timeCol: 时间列
            barReturnCol: 1根Bar的区间收益率(For 分层回测法)
            futureReturnCols: 未来区间收益率列名list(For IC法&回归法)
            returnIntervals: 收益率计算间隔
            dailyFreq: 表示当前因子输入是否为日频, false表示输入分钟频因子回测
            callBackPeriod: 回看周期, 默认为1(即只使用当前period数据进行因子统计量计算)
            quantiles: 分组数量, 每个period中标的会根据当前因子的值从小到大分成quantiles个数的分组去统计分组收益率
            dailyPnlLimit: 当且仅当dailyFreq=true时生效, 表示日涨跌幅限制
            useMinFreqPeriod: 仅当分钟频因子评价时有效, true表示计算因子统计量时按照分钟频聚合计算，false则按照日频聚合计算
            */
            totalData = df
            if (dailyFreq==true or (dailyFreq==false and useMinFreqPeriod==true)){{ // 分钟频->分钟频 & 日频->日频
                // for ICIR & 回归法, 使用原始时间频率生成period
                time_list = sort(distinct(totalData[timeCol]),true) // 分钟时间列/日时间列
                period_dict = dict(time_list, cumsum(take(1, size(time_list))))
                time_dict = dict(values(period_dict), keys(period_dict))
                totalData[`period] = period_dict[totalData[timeCol]]  // timeCol -> period
                period_list = values(period_dict) // 所有period组成的list
        
                // for 分层回测法，与回归法一致
                totalData[`quantilePeriod] = totalData[`period]
                qperiod_list = period_list
                qtime_dict = time_dict
            }}else{{ // 分钟频->日频
                // for 分层回测法, 依然使用原始分钟频生成period
                qtime_list = sort(distinct(totalData[timeCol]),true) // 分钟时间列
                qperiod_dict = dict(qtime_list, cumsum(take(1, size(qtime_list))))
                qtime_dict = dict(values(qperiod_dict), keys(qperiod_dict))
                totalData[`quantilePeriod] = qperiod_dict[totalData[timeCol]]  // timeCol -> qperiod
                qperiod_list = values(qperiod_dict)
        
                // for ICIR & 回归法, 生成日频period
                time_list = sort(distinct(sql(select=sqlColAlias(makeCall(date, sqlCol(timeCol)),"time"), from=totalData).eval()["time"]), true) // 日期时间列
                period_dict = dict(time_list, cumsum(take(1, size(time_list))))
                time_dict = dict(values(period_dict), keys(period_dict))
                totalData[`period] = period_dict[date(totalData[timeCol])]  // timeCol -> period
                period_list = values(period_dict) // 所有period组成的list
            }};

            // 计算不同周期的收益率(这里已经提前计算好了然后把列名传进来了)
            for (i in 0..(size(futureReturnCols)-1)){{
                rename!(totalData, futureReturnCols[i], `Ret+string(returnIntervals[i]))
            }}
            returnCol = `Ret+string(returnIntervals);
            rename!(totalData, barReturnCol, `period_return);
            
            // 分层回测 \ ICIR法&回归法
            sortBy!(totalData, timeCol, 1)            
            if (dailyPnlLimit!=NULL and dailyFreq==true){{
                update totalData set period_return = clip(period_return, -dailyPnlLimit, dailyPnlLimit)
            }}
            colList = returnCol.copy().append!(idCol).append!(timeCol).append!([`period,`quantilePeriod]).append!(factor_list)
            regData = sql(sqlCol(colList),from=totalData).eval()  // for ICIR法 & 回归法
            quantileData = sql(sqlCol(colList).append!(sqlCol(`period_return)), from=totalData).eval() // for 分层回测法
            counter = 0
            for (interval in returnIntervals){{
                print("processing ReturnInterval:"+string(interval))
        
                // 分层回测
                print("Start Quantile BackTesting...")
                QuantileFunc = QuantileStats{{quantileData, idCol, factor_list, interval, quantiles, }} // DolphinDB函数部分化应用
                qes = peach(QuantileFunc, qperiod_list).unionAll(false)
                print("End Quantile BackTesting...")

                // ICIR法&回归法
                RegStatsFunc = RegStats{{regData, factor_list, interval, callBackPeriod, }}; // DolphinDB函数部分化应用
                res = peach(RegStatsFunc, period_list).unionAll(false)

                if (counter == 0){{
                    summary_res = res
                    quantile_res = qes
                }}else{{
                    summary_res.append!(res)
                    quantile_res.append!(qes)
                }}
                counter += 1
            }}
            print("SingleFactor Evaluation End")
            summary_res[`TradeTime] = time_dict[summary_res[`period]]     // 添加时间
            quantile_res[`TradeTime] = qtime_dict[quantile_res[`period]]
            sortBy!(summary_res,[`TradeTime,`factor,`period],[1,1,1])
            sortBy!(quantile_res,[`TradeTime,`factor,`period],[1,1,1])
            return summary_res, quantile_res
        }}
        """

    def summary_command(self):
        """[单因子回测]individual_return(period_return)&summary_result&summary_daily_result"""
        return rf"""  
        // 配置项
        factor_list = {self.factor_list}
        idCol = "{self.idCol}";
        timeCol = "timeForCal"; // concatDateTime
        barReturnCol = "{self.barReturnCol}";
        futureReturnCols = {self.futureReturnCols};
        callBackPeriod = {int(self.callBackPeriod)}
        quantiles = {int(self.Quantile)};
        returnIntervals = {self.returnIntervals};
        if ({int(self.dailyPnlLimit is not None)}==1){{
            dailyPnlLimit = {self.dailyPnlLimit};
        }}else{{
            dailyPnlLimit = NULL;
        }}
        if ({int(self.dailyFreq)}==1){{
            dailyFreq = true;
        }}else{{
            dailyFreq = false;
        }}
        if ({int(self.useMinFreqPeriod)}==1){{
            useMinFreqPeriod = true;
        }}else{{
            useMinFreqPeriod = false;
        }}
        
        // 获取数据
        pt = select *, concatDateTime(TradeDate,TradeTime) as `timeForCal
            from loadTable("{self.combineDB}","{self.combineTB}") order by symbol,TradeDate,TradeTime;
        
        // 执行单因子评价
        summary_res, quantile_res = SingleFactorAnalysis(pt, factor_list, idCol, timeCol, barReturnCol, futureReturnCols,
         returnIntervals, dailyFreq, callBackPeriod=callBackPeriod, quantiles=quantiles, 
            dailyPnlLimit=dailyPnlLimit, useMinFreqPeriod=useMinFreqPeriod)
        
        // 插入至数据库
        InsertData(DBName="{self.resultDB}", TBName="{self.summary_table}", 
                            data=summary_res, batchsize=1000000);
        print("IC法&回归法结果插入完毕")
        InsertData(DBName="{self.resultDB}", TBName="{self.quantile_table}", 
                            data=quantile_res, batchsize=1000000);
        print("分层回测法结果插入完毕")
        undef(`summary_res`quantile_res); // 释放内存
        """

    def BackTest(self):
        # 因子模型部分
        self.init_ResultDataBase(dropTB=True)
        self.session.run(self.init_def())
        self.session.run(self.summary_command())

if __name__=="__main__":
    from src.factor import CombineFunc,Label
    session=ddb.session()
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")

    with open(r".\config\factorEva_cfg.json5", mode="r", encoding="UTF-8") as file:
        cfg = json5.load(file)
    F=SingleFactorBackTest(
        session=session,pool=pool,config=cfg,
        Label_prepareFunc=[Label.get_DayBarLabel,
                           Label.get_DayLabel1,
                           Label.get_DayLabel5,
                           Label.get_DayLabel10,
                           Label.get_DayLabel20],
        Combine_prepareFunc=CombineFunc.Combine,
    )
    # 初始化标签数据库
    # F.init_labelDB(dropDB=False,dropTB=True)
    # F.add_LabelData()
    # # 如果原始数据没有变化，那么不用运行init_CombineDatabase()与add_CombineData()
    F.init_combineDB(dropDB=True)
    F.add_CombineData()
    F.BackTest()