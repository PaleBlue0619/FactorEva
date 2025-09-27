import os
import dolphindb as ddb
from src.FactorEva import SingleFactorBackTest

def Combine(self: SingleFactorBackTest):
    """
    日频&分钟频标签-因子值Combine函数
    """
    self.session.run(rf"""
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
    """)

    # 拿数据
    # idx_code = "399300.SZ" # 300
    idx_code = "000905.SH"  # 500
    # idx_code = "000852.SH" # 1000
    self.session.run(rf"""
    // 最后的数据格式: symbol,TradeDate,TradeTime,BarReturn,FutureReturn..,factor_list
    idx_code = "{idx_code}"
    label_list = {[self.barReturnCol]+self.futureReturnCols}
    factor_list = {self.factor_list};
    symbol_list = exec distinct(con_code) as component 
                from loadTable("dfs://DayKDB","o_tushare_index_weight") 
                where index_code == idx_code and 
                (trade_date between {self.start_dot_date} and {self.end_dot_date});
    
    // 指数数据index_df
    index_df = select con_code as symbol, trade_date as TradeDate
                from loadTable("dfs://DayKDB","o_tushare_index_weight")
                where index_code == idx_code and (trade_date between {self.start_dot_date} and {self.end_dot_date});
    total_date_list = getMarketCalendar("XSHG", date({self.start_dot_date}), 
                        temporalAdd(date({self.end_dot_date}), 1 ,"XSHG"))
    current_date_list = sort(exec distinct(TradeDate) from index_df);
    last_date = current_date_list[0]
    for (i in 1..size(total_date_list)-1){{
        ts = total_date_list[i]
        if (!(ts in current_date_list)){{
            // 离他最近比它小的date
            index_df.append!(select symbol, ts as `TradeDate from index_df where TradeDate == last_date)      
        }}else{{
            last_date = ts
        }}
    }}
    update index_df set index_state = 1.0;
    sortBy!(index_df,`TradeDate`symbol,[1,1]);
    
    // 标签数据label_df
    label_df = select label from loadTable("{self.labelDB}","{self.labelTB}")
            where labelName in label_list and (TradeDate between {self.start_dot_date} and {self.end_dot_date})
            pivot by symbol, TradeDate, TradeTime, labelName;
    label_df = sql(select = sqlCol(["symbol","TradeDate","TradeTime"].append!(label_list)),
                    from = label_df).eval() // 整理顺序
    
    // 因子数据factor_df
    if ({int(self.dailyFreq)}==1){{
        factor_df = select * from loadTable("{self.factorDB}","{self.factorTB}") 
                where factor in {self.factor_list} and (date between {self.start_dot_date} and {self.end_dot_date});
        update factor_df set value = nullFill(value, avg(value)) context by factor,date // 截面均值填充
        factor_df = select value from factor_df pivot by symbol,date as TradeDate,factor
        update factor_df set TradeTime = 15:00:00.000;
    }}else{{
        factor_df = select * from loadTable("{self.factorDB}","{self.factorTB}") 
                where factor in {self.factor_list} and (date between {self.start_dot_date} and {self.end_dot_date})
        update factor_df set value = nullFill(value, avg(value)) context by factor,date,time // 截面均值填充
        factor_df = select value from factor_df pivot by symbol,date as TradeDate,time as TradeTime,factor
    }}
    factor_df = sql(select = sqlCol(["symbol","TradeDate","TradeTime"].append!(factor_list)),
                    from = factor_df).eval() // 整理顺序
    
    // left Join
    totalData = select * from lsj(label_df, index_df, `TradeDate`symbol) where index_state == 1.0;
    dropColumns!(totalData, `index_state);
    totalData = lsj(totalData, factor_df, `TradeDate`TradeTime`symbol);
    undef(`factor_df`label_df); // 释放内存

    // 添加period
    update totalData set timeForCal = concatDateTime(TradeDate,TradeTime);
    time_list = sort(exec distinct(timeForCal) from totalData, true);
    time_dict = dict(time_list, 1..size(time_list));
    totalData[`period] = int(time_dict[totalData[`timeForCal]]);
    
    // 添加至模板数据库
    template_pt = select first(timeForCal.date()) as startDate,
                        first(timeForCal.time()) as startTime,
                        last(timeForCal.date()) as endDate,
                        last(timeForCal.time()) as endTime
                from totalData group by period;
    loadTable("{self.combineDB}","{self.template_table}").append!(template_pt);
    template_ind = select first(timeForCal.date()) as TradeDate,
                        first(timeForCal.time()) as TradeTime
                    from totalData group by symbol, period;
    loadTable("{self.combineDB}","{self.template_individual_table}").append!(template_ind);
    
    // 最终数据
    totalData = sql(select=sqlCol(`symbol`TradeDate`TradeTime).append!(sqlCol(label_list)).append!(sqlCol(factor_list)).append!(sqlCol(`period)),
                    from=totalData).eval()
    InsertData("{self.combineDB}", "{self.combineTB}", totalData, batchsize=1000000);
    
    undef(`totalData); // 清除缓存
    """)
