import pandas as pd
import dolphindb as ddb
from src.FactorEva import OptimizeFactorAttribute

def get_AttributeData(self: OptimizeFactorAttribute):
    """
    Factor Attribute准备函数
    """
    weightName = "testWeight"
    self.session.run(f"""
    // Basic Config
    startDate = {self.start_dot_date};
    endDate = {self.end_dot_date};
    idx = "000905.SH"
    
    // 准备Mod, 用于left semi join
    mod = select symbol,TradeDate as date from loadTable("{self.weightDB}","{self.weightTB}")
            where TradeDate between startDate and endDate;
    symbolList = exec distinct symbol from mod;
    dateList = sort(exec distinct date from mod); // 所有日期列表
    
    // 1.Benchmark
    bench = select trade_date as date, con_code as symbol, weight/sum(weight) as weight
        from loadTable("dfs://DayKDB","o_tushare_index_weight")
        where index_code == idx and (trade_date between startDate and endDate)
        context by trade_date 
        order by trade_date;
    bench = select * from lsj(mod.copy(), bench, `date`symbol);
    update bench set weight = weight.ffill() context by symbol;
    update bench set weight = nullFill(weight/sum(weight),0.0) context by date;
    update bench set weight = 1\count(symbol) where weight == 0.0 context by date;
    share(bench,"{self.benchPosObj}"); 
    print("bench has been prepared!")
        
    // 2.Own
    own = select TradeDate as date,symbol, weight
        from loadTable("{self.weightDB}","{self.weightTB}") 
        where weightName == "{weightName}" and (TradeDate between startDate and endDate)
    own = select * from lsj(mod.copy(),own,`date`symbol);
    update own set weight = nullFill!(weight,0.0);
    share(own,"{self.ownPosObj}");
    print("own has been prepared!")
    
    // 3. Market
    market = select TradeDate as date,symbol,label as marketReturn
        from loadTable("{self.labelDB}","{self.labelTB}")
        where labelName == "ret1D" and (TradeDate between startDate and endDate);
    market = select * from lsj(mod.copy(),market,`date`symbol);
    update market set marketReturn = nullFill!(marketReturn,0.0);
    share(market,"{self.returnObj}")
    print("market has been prepared!")
        
    // 4. styleExpos
    styleExpos = select value
        from loadTable("{self.factorDB}","{self.factorTB}") 
        where factor in {self.factor_list} and (date between startDate and endDate) 
        pivot by date,symbol,factor
    styleExpos = select * from lsj(mod.copy(),styleExpos,`date`symbol)
    styleExpos = nullFill!(styleExpos, 0.0);
    share(styleExpos,"{self.styleFactorObj}");
    print("styleExpos has been prepared!")
    
    // 5. industryExpos
    industryExpos = select ts_code as symbol,l1_name as industry from 
            loadTable("dfs://DayKDB","o_tushare_sw_industry_member")
            context by ts_code limit -1 // 开上帝视角,只取最新的行业分布
    industryExpos = select * from lsj(mod.copy(),industryExpos,`symbol);
    industryList = exec distinct industry from industryExpos;
    industryExpos = oneHot(industryExpos,`industry);
    newIndustryList = "industry_"+industryList.copy()
    industryExpos = <select date,symbol,_$$newIndustryList from industryExpos>.eval()
    share(industryExpos,"{self.industryFactorObj}");
    print("industryExpos has been prepared!")
    """)
