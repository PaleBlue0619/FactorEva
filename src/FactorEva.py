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

class FactorBackTest:
    """单因子评价与组合优化回测的共同逻辑&共同建库建表函数"""
    def __init__(self, session: ddb.session, pool:
                ddb.DBConnectionPool, config: Dict,
                 ):
        # 基本信息
        self.strategy_name="strategy"
        self.session=session
        self.pool=pool
        self.factorDB = config["factor_database"]
        self.factorTB = config["factor_table"]
        self.labelDB = config["label_database"]
        self.labelTB = config["label_table"]
        self.resultDB = config["result_database"]
        self.weightDB = ""
        if "weight_database" in config.keys():
            self.weightDB = config["weight_database"]
        self.weightTB = ""
        if "weight_table" in config.keys():
            self.weightTB = config["weight_table"]

        # 变量类
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.start_dot_date = pd.Timestamp(self.start_date).strftime('%Y.%m.%d')
        self.end_dot_date = pd.Timestamp(self.end_date).strftime('%Y.%m.%d')
        self.factor_list = config["factor_list"]
        self.dailyFreq = True
        if "dailyFreq" in config.keys():
            self.dailyFreq = config["dailyFreq"]  # 是否为日频

    def get_factorList(self, inplace: bool = True) -> Dict:
        """
        自动解析用户输入的dbName+tbName对应的因子, 输出一个字典
        数据格式:
        """
        factor_df = self.session.run(f"""
            select count(*) from loadTable("{self.factorDB}","{self.factorTB}") group by factor
        """)
        factor_list = factor_df["factor"].tolist()
        if inplace:
            self.factor_list = factor_list
        return factor_list

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

    def init_weightDB(self, dropDB: bool = False, dropTB: bool = False):
        """
        初始化权重数据库
        """
        if dropDB and self.session.existsDatabase(self.weightDB):
            self.session.dropDatabase(self.weightDB)
        if dropTB and self.session.existsTable(dbUrl=self.weightDB, tableName=self.weightTB):
            self.session.dropTable(dbPath=self.weightDB, tableName=self.weightTB)
        if not self.session.existsTable(self.weightDB, self.weightTB):
            """新建数据库表"""
            self.session.run(f"""
                db = database("{self.weightDB}",VALUE,2010.01M..2030.01M, engine="TSDB")
                schemaTb = table(1:0, ["symbol","TradeDate","TradeTime","weightName","weight"],
                [SYMBOL,DATE,TIME,STRING,DOUBLE])
                db.createPartitionedTable(schemaTb, "{self.weightTB}", partitionColumns=`TradeDate, 
                    sortColumns=`weightName`symbol`TradeTime`TradeDate, keepDuplicates=LAST)
            """)

class SingleFactorBackTest(FactorBackTest):
    def __init__(self, session: ddb.session, pool: ddb.DBConnectionPool, config: Dict,
        Label_prepareFunc=None,
        Combine_prepareFunc=None
    ):
        super(SingleFactorBackTest, self).__init__(session, pool, config)

        # 库表类
        self.combineDB = config["combine_database"]
        self.combineTB = config["combine_table"]
        self.quantile_table= config["quantile_table"]      # 分层回测结果表
        self.summary_table= config["summary_table"]        # ICIR&回归法结果表

        # 函数类
        self.Label_prepareFunc = Label_prepareFunc  # 标签准备函数
        self.Combine_prepareFunc=Combine_prepareFunc # 数据合并函数

        # 变量类
        self.returnIntervals = config["returnIntervals"] # [1,5,10,20,30] 区间收益率长度
        self.callBackPeriod = int(config["callBackPeriod"])  # 回看长度
        self.Quantile=int(config["quantile"])  # 因子分组收益率统计
        self.dailyPnlLimit = config["dailyPnlLimit"]
        self.useMinFreqPeriod = config["useMinFreqPeriod"]
        self.idCol = config["idCol"]
        self.barReturnCol = config["barReturnLabel"]
        self.futureReturnCols = config["futureReturnLabels"]

        # 中间计算+最终结果类
        self.template_table="template"
        self.template_individual_table="template_individual"

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

    def add_CombineData(self):
        self.Combine_prepareFunc(self)

    def add_labelData(self):
        if isinstance(self.Label_prepareFunc, list):    # 如果有多个标签函数
            for labelFunc in self.Label_prepareFunc:
                labelFunc(self)
        else:   # 如果只传入一个标签函数
            self.Label_prepareFunc(self)

    def init_resultDB(self, dropTB=False):
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
        """单因子回测框架"""
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

    def run(self):
        # 因子模型部分
        self.init_resultDB(dropTB=True)
        self.session.run(self.init_def())
        self.session.run(self.summary_command())

class OptimizeFactorBackTest(FactorBackTest):
    def __init__(self, session:ddb.session, pool:ddb.DBConnectionPool, config:Dict, Optimize_callBackFunc):
        super(MultiFactorOptimize, self).__init__(session, pool, config)
        self.strategy_name = "optimize_strategy"
        self.session = session
        self.pool = pool

        # 变量类
        self.callBackPeriod = int(config["callBackPeriod"]) # 回看长度
        self.callBackInterval = int(config["callBackInterval"]) # 回看频率
        self.retFactor = "shio" # 预期收益率因子
        self.periodDF = "optPeriodDF"
        self.freq = "d" if config["dailyFreq"] else "m"
        self.optData = pd.DataFrame()

        # 回调类
        self.currentDate = pd.Timestamp(self.start_date)
        self.currentTime = 1500
        self.currentPeriod = 1

        # 函数类
        self.Optimize_callBackFunc= Optimize_callBackFunc

    def init_resultDB(self, dropTB: bool = False):
        if dropTB:
            if self.session.existsTable(dbUrl=self.resultDB, tableName=self.optimize_table):
                self.session.dropTable(dbPath=self.resultDB, tableName=self.optimize_table)
        if not self.session.existsTable(self.resultDB, self.optimize_table):
            columns_name = ["weightName","TradeDate","TradeTime","return"]
            columns_type = ["SYMBOL", "DATE", "TIME", "DOUBLE"]
            self.session.run(f"""
                       db=database("{self.resultDB}");
                       schemaTb=table(1:0,{columns_name},{columns_type});
                       t=db.createDimensionTable(table=schemaTb,tableName="{self.optimize_table}")
            """)

    def get_periodList(self):
        """
        返回一个共享变量->TradeDate TradeTime period
        最终格式: symbol TradeDate TradeTime period label factor
        共享变量: TradeDate TradeTime MaxDate MaxTime period
        """
        self.session.run(f"""
            freq = "{self.freq.lower()}"
            if (freq == "d"){{ // 日频
                pt = select 15:00:00.000 as TradeTime, 1.0 as period 
                     from loadTable("{self.factorDB}","{self.factorTB}") 
                     where date between {self.start_dot_date} and {self.end_dot_date}
                    group by date as TradeDate
            }}else{{ // 分钟频
                pt = select 1.0 as period from loadTable("{self.factorDB}","{self.factorTB}")
                    where date between {self.start_dot_date} and {self.end_dot_date}
                    group by date as TradeDate, time as TradeTime
            }}
            update pt set period = cumsum(period);
            share(pt, "{self.periodDF}");  // 共享变量
        """)

    def getData(self, startPeriod: int, endPeriod: int, factorList: list) -> pd.DataFrame:
        """
        返回一张宽表: symbol,TradeDate,TradeTime,label,factorList
        :param labelName: 标签名称str
        :param startPeriod: 开始的period int
        :param endPeriod: 结束的period int
        :param factorList: Barra因子列
        :return:
        """
        if not factorList:
            factorList = []

        if startPeriod == endPeriod:
            # 说明只需要取一个Period的factor
            data = self.session.run(rf"""
                // 获取时间频率
                freq = "{self.freq.lower()}"

                // 获取这个period对应的TradeDate & TradeTime
                DF = select * from objByName("{self.periodDF}", true)
                idx = find(DF[`period],{startPeriod})
                targetDate = DF[`TradeDate][idx]
                targetTime = DF[`TradeTime][idx]

                // 获取特征列名称
                featureList = {factorList};
                // 从因子数据库&标签数据库中获取数据
                if (freq=="d"){{    // 日频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                where factor in featureList and date == targetDate 
                                pivot by symbol, date as TradeDate, factor
                    // update factor_df set TradeTime = 15:00:00.000     
                }}else{{    // 分钟频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                where factor in featureList and date == targetDate and time == targetTime
                                pivot by symbol, date as TradeDate, time as TradeTime, factor
                }}
                factor_df
            """,disableDecimal=True)
            return data

        if startPeriod < endPeriod:
            # 说明需要取一个区间的数据
            data = self.session.run(rf"""
                // 获取时间频率
                freq = "{self.freq.lower()}"

                // 获取这个period对应的TradeDate & TradeTime
                DF = select * from objByName("{self.periodDF}", true)
                startIdx = find(DF[`period],{startPeriod})
                endIdx = find(DF[`period],{endPeriod})
                startDate = DF[`TradeDate][startIdx]
                endDate = DF[`TradeDate][endIdx]
                startTime = DF[`TradeTime][startIdx]
                endTime = DF[`TradeTime][endIdx]
                startTimeStamp = concatDateTime(startDate,startTime)
                endTimeStamp = concatDateTime(endDate,endTime)

                // 获取特征列名称
                featureList = {factorList};
                // 从因子数据库&标签数据库中获取数据
                if (freq=="d"){{    // 日频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}") 
                                where factor in featureList and (date between startDate and endDate) 
                                pivot by symbol, date as TradeDate, factor
                    // update factor_df set TradeTime = 15:00:00.000
                }}else{{    // 分钟频
                    // 因子数据
                    factor_df = select value from loadTable("{self.factorDB}","{self.factorTB}")
                                where factor in featureList and (concatDateTime(date,time) between startTimeStamp and endTimeStamp)
                                pivot by symbol, date as TradeDate, time as TradeTime, factor
                }}
                factor_df
            """,disableDecimal=True)
            return data

    def run(self):
        """组合优化回测框架"""
        periodDF = self.session.run(f"""
            select TradeDate,TradeTime,period from objByName("{self.periodDF}",true);
        """)
        dateList = periodDF["TradeDate"].tolist()
        timeList = periodDF["TradeTime"].tolist()
        periodList = periodDF["period"].tolist()
        factorList = list(set([self.retFactor]+self.factor_list))

        weightAppender = ddb.PartitionedTableAppender(dbPath=self.weightDB,
                                                      tableName=self.weightTB,
                                                      partitionColName="TradeDate",
                                                      dbConnectionPool=pool)
        weight = pd.DataFrame()
        for i in tqdm.tqdm(range(0, periodDF.shape[0]),desc="Optimize BackTesting..."):
            self.currentDate = dateList[i]
            self.currentTime = timeList[i]
            self.currentPeriod = periodList[i]
            if self.currentPeriod % self.callBackInterval == 0 and i>=self.callBackInterval:
                self.optData = self.getData(startPeriod=max(1,self.currentPeriod+1-self.callBackPeriod),
                                    endPeriod=self.currentPeriod,
                                    factorList=factorList)
                weight = self.Optimize_callBackFunc(self)
            # 这里相当于每期都要append, 但是并不是每期都更新
            if not weight.empty:
                weightAppender.append(weight)

class OptimizeFactorAttribute(FactorBackTest):
    def __init__(self, session: ddb.session,
                 pool: ddb.DBConnectionPool,
                 config: Dict,
                 Data_prepareFunc):
        super(OptimizeFactorAttribute, self).__init__(session, pool, config)
        # 共享变量名称
        self.benchPosObj = "OptAttrBenchPos"
        self.ownPosObj = "OptAttrOwnPos"
        self.returnObj = "OptAttrReturn"
        self.styleFactorObj = "OptAttrStyleFactor"
        self.industryFactorObj = "OptAttrIndustryFactor"

        # 变量类
        self.windowSize = int(config["windowSize"])

        # 函数类
        self.Data_prepareFunc = Data_prepareFunc

    def init_def(self):
        self.session.run("""
        /*z-score标准化
         *按日对横截面因子归一处理
        */
        def zscoreStyleExpos(styleExpos){
            styleCols = columnNames(styleExpos)[2:]
            zscoreTb = <select date, symbol, zscore(_$$styleCols) as _$$styleCols from styleExpos context by date>.eval() 
            return zscoreTb 
        }
        
        /*processFactor--求解纯因子收益率f_j = w_j**r -- 用于each并行
         *输入：j--风格因子，q--风格因子个数，p--行业因子个数
                Aeq--包含了所有因子暴露矩阵：维度（1+p+q）*n 
                beq--包含了所有因子暴露度，维度 1*（1+p+q） 
                lb--下界：保证权重系数不为负数  
                ub--上界：该问题中不做约束
                retVec:股票收益率向量 1*n
         *输出：纯因子j的收益率  f_j
         */
        def processFactor(j, q, p, Aeq, beq, retVec, lb, ub){
                    //print("开始求解：第"+j+"个因子")
                    X_T = Aeq[1:(q+1),]     //T日的风格因子暴露矩阵转置：q*n维度 
                    H_T = Aeq[q+1:,]      //行业因子暴露矩阵转置：p*n维度 
                    c = -flatten(Aeq[j+1,])    
                    solved = false   //是否有解
                    //等式约束求解
                    try{
                        beq_neq = beq
                        beq_neq[1 + j] = 1.0    // 目标因子 j 曝光 = 1
                        w_j = linprog(c, , , Aeq, beq_neq, lb, ub)[1]
                        return  dot(w_j, retVec)
                    }catch(ex){}
         
                    //初始不等式约束阈值
                    threshold = 0.00  
                    targetThreshold = 1.0  
                
                    // 使用 do-while 逐步放宽约束 
                    do {  
                        //print("求解第"+j+"个因子："+targetThreshold+":"+threshold)
                        // 初始化不等式约束矩阵 A 和向量 b  
                        A_ineq = matrix(DOUBLE, 0, retVec.size())  
                        b_ineq = array(DOUBLE, 0)  
                        Aeq_cu = concatMatrix([Aeq[0,:],H_T], false) 
                        beq_cu = join(beq[0], beq[1+q:])  
                        
                        //目标因子不等式约束    -X_j <= -targetThreshold  X_j <= 2-targetThreshold
                         if(targetThreshold == 1.0 and threshold > 0){
                            A_ineq = concatMatrix([A_ineq, X_T[:j,], X_T[j+1:,], -X_T[:j,], -X_T[j+1:,]], false)   
                            b_ineq.append!(take(threshold,2*q-2))  
                            Aeq_current = concatMatrix([Aeq_cu, X_T[j,]], false)
                            beq_current = join(beq_cu, beq[1+j])
                         }else if(targetThreshold < 1.0 and threshold == 0) {  
                            A_ineq = concatMatrix([A_ineq, -X_T[j,:], X_T[j,:]], false)   
                            b_ineq.append!(-targetThreshold).append!(2-targetThreshold)  
                            Aeq_current = concatMatrix([Aeq[:(1+j),:], Aeq[(2+j):,:]], false)  
                            beq_current = join(beq[:(1+j)], beq[(2+j):])  
                        }else{
                            A_ineq = concatMatrix([A_ineq, -X_T[j,:], X_T[j,:], X_T[:j,], X_T[j+1:,], -X_T[:j,], -X_T[j+1:,]], false)   
                            b_ineq.append!(-targetThreshold).append!(2-targetThreshold) 
                            b_ineq.append!(take(threshold, 2*q-2))  
                            Aeq_current = Aeq_cu
                            beq_current = beq_cu
                        }
                  
                        //尝试求解，逐步放宽不等式约束条件
                        try{  
                            w_j = linprog(c, A_ineq, b_ineq, Aeq_current, beq_current, lb, ub)[1]   
                            solved = true  
                            return  dot(w_j, retVec)
                        }catch(ex) {  
                            if(threshold < 0.5) {  
                                threshold += 0.05  // 继续放宽非目标因子约束  
                            } else if(targetThreshold > 0.5) {   
                                // 非目标因子约束已达到最大值，开始放宽目标因子约束  
                                    targetThreshold -= 0.05  // 从0.95开始逐步降低  
                                    threshold = 0.00  // 重置非目标因子约束  
                                }else {  
                                    // 最终降级：只保留权重和与行业约束  
                                    targetThreshold = 0
                                    threshold = 0
                                }  
                            }    
                    }while(!solved)
                }
        
        
        /*计算某日的纯因子收益率--用于 peach 并行 */
        def processdailyFactor(day, exposTb,styleCols,industryCols,n,p,q){
            dailyExpos = select * from exposTb where date = day
            X = matrix(<select _$$styleCols from dailyExpos>.eval())      //T日的风格因子暴露矩阵
            H = matrix(<select _$$industryCols from dailyExpos>.eval())   //行业因子哑变量矩阵
            w0 = dailyExpos.weight
            retVec = dailyExpos.marketReturn 
            h = dot(w0, H)    //基准对应的行业权重向量
            rowSumMat = matrix(take(1.0, n)).transpose() 
            Aeq = concatMatrix([rowSumMat, X.transpose(), H.transpose()],false)
            beq = take(0.0, 1 + q + p)  
            beq[0] = 1.0    // ∑w = 1  
            beq[1+q:] = h    // 行业敞口保持基准 
            f_vector = each(processFactor{, q, p, Aeq, beq, retVec, 0.0, 1.0} , 0..(q-1)) 
            return f_vector
        }
        
        
        /*pureFactorReturn--纯因子投资组合估计函数
         *输入：dateList--回测时间(交易日)
                bench--市场基准信息
                market--个股收益率
                styleExpos--风格因子暴露表
                industryExpos--行业因子暴露表
         *输出：纯因子投资组合收益率 f_j；所有因子暴露矩阵 X
         */
        def pureFactorReturn(dateList, bench, market, styleExpos, industryExpos){
            styleCols = columnNames(styleExpos)[2:]     //获取风格因子以及行业因子的列名
            industryCols = columnNames(industryExpos)[2:] 
            q = size(styleCols)     //因子个数
            p = size(industryCols)    
            n = distinct(styleExpos.symbol).size()
            //初始化结果表 `date`style1`style2,... 列=日期+各因子
            purefactorTb = table(1:0,[`date], [`DATE])
            addColumn(purefactorTb, styleCols, take(DOUBLE, q))
        
            //构建因子暴露矩阵，因子暴露度，个股收益率 retVec
            exposTb = lj(bench, styleExpos,`date`symbol)
            exposTb = lj(exposTb, industryExpos, `date`symbol)
            exposTb = lj(exposTb, market,`date`symbol) 
            //按日循环，获取每日的纯风格因子收益率
            result = peach(processdailyFactor{, exposTb, styleCols, industryCols, n, p, q}, dateList)
            go
            try{re = table(columnNames(result) as date, matrix(result).transpose())
                     purefactorTb.append!(re)}catch(ex){}
            purefactorTb = purefactorTb.sortBy!(`date)
            return purefactorTb, exposTb
        
        }
        
        
        /*计算某日的主动收益因子贡献、残差--用于 peach并行
         */ 
        def processdailyAttRes(day, AllInfoTb, exposTb, styleCols, purefactorTb){
                dailyTb = select * from AllInfoTb where date = day
                X = matrix(<select _$$styleCols from exposTb where date = day>.eval()) // n×q
                rVec = dailyTb.marketReturn    
                //超额主动收益，超额因子敞口
                w_A = dailyTb.weight - dailyTb.own_weight 
                r_A   = dot(w_A, rVec)
                X_A    = dot(w_A, X)  // 1×q
                fVec   = matrix(<exec _$$styleCols from purefactorTb where date=day>.eval()) // 1xq
                contri = X_A * fVec                    // 元素乘，得到 1×q 向量
                mu_A  = r_A - sum(flatten(contri)) 
                return r_A, X_A, contri, mu_A
        }
        
        /*performanceAttribution--主动收益归因函数
         * 输入：dateList--回测时间（交易日）
                own--个股持仓权重
                exposTb--所有因子暴露矩阵 X
                purefactorTb--纯因子投资组合收益率
         *输出：因子贡献残差表factorAttResTb, 主动因子暴露，组合收益表activeExposReturnTb
         */
        def performanceAttribution(dateList, own, exposTb, purefactorTb){
            styleCols = columnNames(purefactorTb)[1:]    // 风格因子列名
            //purefactorTb.rename!(flatten([`date, string(styleCols) + "_purefactor"]))
            q = size(styleCols) 
            n = purefactorTb.size()
            //初始化结果表 `date`style1`style2,...,`residual 列=日期+各因子主动贡献+残差
            factorAttResTb = table(1:0,[`date], [`DATE])    //因子贡献以及残差表
            addColumn(factorAttResTb, styleCols, take(DOUBLE, q))    //主动暴露表 
            activeExposReturnTb = factorAttResTb   //主动暴露表 
            activeExposReturnTb.rename!(flatten([`date, string(styleCols) + "_expos"]))
            AllInfoTb = lj(exposTb, own,`date`symbol)
            acReturn = array(DOUBLE, 0)    //每日主动收益值
            result = peach(processdailyAttRes{, AllInfoTb, exposTb, styleCols, purefactorTb}, dateList)
            go
            acReturn = result.transpose()[0]    
            mu_A = result.transpose()[3]
            X_A = result.transpose()[1]
            Xa = X_A.flatten()$n:q
            activeExposData = table(dateList as date, Xa)
            activeExposReturnTb.append!(activeExposData)
            contri = result.transpose()[2]
            con = contri.flatten()$n:q
            resultData = table(dateList as date, con)
            factorAttResTb.append!(resultData)
            factorAttResTb[`residual] = mu_A
            activeExposReturnTb[`activeReturn] = acReturn 
            return factorAttResTb, activeExposReturnTb
        }
        
        
        /*计算某因子的方差、相关性--用于each 并行 */
        def processStdCorr(j, purefactorTb, activeExposReturnTb, acReturn, windowSize){
               f_j = purefactorTb[columnNames(purefactorTb)[j]]
               X_j = activeExposReturnTb[columnNames(activeExposReturnTb)[j]]
               sigma_f = mstd(f_j, windowSize)
               rho_f = mcorr(f_j, acReturn, windowSize) 
               risk_j = X_j * sigma_f * rho_f
               return risk_j
        }
        
        
        /*riskAttribution--主动风险归因函数
         *输入：dateList--回测时间（交易日）
                purefactorTb--纯因子投资组合收益率
                factorAttResTb--因子贡献残差表
                activeExposReturnTb--主动因子暴露，组合收益表
                windowSize--计算波动率窗口大小
         *输出：因子风险归因表 riskTb
         */
        def riskAttribution(dateList, purefactorTb, factorAttResTb, activeExposReturnTb, windowSize){ 
            styleCols = columnNames(purefactorTb)[1:]
            q = size(styleCols)
        
            resTb = select date, residual, activeReturn from lj(activeExposReturnTb, factorAttResTb,`date)
            acReturn = resTb.activeReturn
            result = each(processStdCorr{, purefactorTb, activeExposReturnTb, acReturn, windowSize}, 1..q)
            //计算残差部分方差、相关性
            sigmaRes = mstd(resTb.residual, windowSize)
            rhoRes = mcorr(resTb.residual, resTb.activeReturn, windowSize)
             //因子主动风险归因表
            riskTb = table(dateList as date, result)
            riskTb.rename!(flatten([`date, string(styleCols)]))
            riskTb[`residualRisk] = sigmaRes* rhoRes
        
            newCols = columnNames(riskTb)[1:]
            tb = <select sum(_$$newCols) as totalRisk from riskTb>.eval()
            riskTb[`totalRisk] = tb.totalRisk   //增加一列风险和
            return riskTb[windowSize-1:,]
            }
        
        //归因分析：
        /*输入：startDate, endDate---回测时间
                windowSize---波动率、相关性窗口
                bench--市场基准信息
                own--股票组合持仓信息
                market--股票收益表
                styleExpos--风格因子暴露表
                industryExpos--行业因子暴露表
                standard--是否进行因子归一化
         *输出：纯因子收益率表，主动收益归因表，主动风险归因表
         */
        def attributionFunc(startDate, endDate, bench, own, market, styleExpos, industryExpos, windowSize, standard=false){
            bench_ = select * from bench where date >= startDate and date<= endDate
            market_ = select * from market where date >= startDate and date<= endDate
            styleExpos_ = select * from styleExpos where date >= startDate and date<= endDate
            industryExpos_ = select * from industryExpos where date >= startDate and date<= endDate
            dateList = exec distinct date from market_ where date >= startDate and date<= endDate  //筛选出交易日
            styleExposNew = iif(standard, zscoreStyleExpos(styleExpos_), styleExpos_)
            purefactorTb, exposTb  = pureFactorReturn(dateList, bench_, market_, styleExposNew, industryExpos_)
            factorAttResTb, activeExposReturnTb = performanceAttribution(dateList, own, exposTb, purefactorTb)
            riskTb = riskAttribution(dateList, purefactorTb, factorAttResTb, activeExposReturnTb, windowSize)
            return dict(`纯因子收益率`主动收益归因`主动风险归因, (purefactorTb, factorAttResTb, riskTb), true)
        }
        
        """)

    # def init_resultDB(self, dropDB: bool=False, dropTB:bool=True):

    def run(self):
        # 生成共享变量
        self.Data_prepareFunc(self)
        self.init_def() # 初始定义相关函数
        self.session.run(f"""
            // Basic Config
            startDate = {self.start_dot_date};
            endDate = {self.end_dot_date};
            windowSize = {self.windowSize};
            standard = true;
            
            // 反射获取相关共享变量
            bench = objByName("{self.benchPosObj}",true); // 市场基准权重
            own = objByName("{self.ownPosObj}",true); // 每日个股实际持仓权重表
            market = objByName("{self.returnObj}",true); // 每日个股收益率
            styleExpos = objByName("{self.styleFactorObj}",true); // 每日风格因子暴露表
            industryExpos = objByName("{self.industryFactorObj}",true); // 每日行业因子哑变量表
            
            // 调用计算函数
            resDict = attributionFunc(startDate, endDate, bench, own, market, styleExpos,
                  industryExpos, windowSize, standard);
            // 返回键为: "纯因子收益率"/"主动收益归因"/"主动风险归因"
        """)

if __name__=="__main__":
    from src.factor import CombineFunc,Label,OptimizeFunc,AttributeFunc
    session=ddb.session()
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")

    # with open(r".\config\factorEva_cfg.json5", mode="r", encoding="UTF-8") as file:
    #     cfg = json5.load(file)
    # F = SingleFactorBackTest(
    #     session=session, pool=pool, config=cfg,
    #     Label_prepareFunc=[Label.get_DayBarLabel,
    #                         Label.get_DayLabel1,
    #                         Label.get_DayLabel5,
    #                         Label.get_DayLabel10,
    #                         Label.get_DayLabel20],
    #     Combine_prepareFunc=CombineFunc.Combine,
    # )
    # # 初始化标签数据库
    # # F.init_labelDB(dropDB=False,dropTB=True)
    # # F.add_labelData()
    # # # 如果原始数据没有变化，那么不用运行init_CombineDatabase()与add_CombineData()
    # F.init_combineDB(dropDB=True)
    # F.add_CombineData()
    # F.run()

    # with open(r".\config\factorOpt_cfg.json5", mode="r", encoding="UTF-8") as file:
    #     cfg = json5.load(file)
    # W = OptimizeFactorBackTest(session=session, pool=pool, config=cfg,
    #                         Optimize_callBackFunc=OptimizeFunc.myOptimizeFunc)
    # # W.init_weightDB(dropDB=False, dropTB=True)
    # # W.init_resultDB(dropTB=True)
    # W.get_periodList()
    # W.run()

    with open(r".\config\factorAttr_cfg.json5", mode="r", encoding="UTF-8") as file:
        cfg = json5.load(file)
    A = OptimizeFactorAttribute(session=session, pool=pool, config=cfg,
                                Data_prepareFunc=AttributeFunc.get_AttributeData)
    A.run()