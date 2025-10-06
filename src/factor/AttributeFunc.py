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
        where labelName == "barReturnCol" and (TradeDate between startDate and endDate);
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

def get_AttributeTest(self: OptimizeFactorAttribute):
    """For test Data"""
    self.session.run(f"""
    //模拟数据：因子归因
    startDate = {self.start_dot_date}
    endDate = {self.end_dot_date}	 //1年
    windowSize = {self.windowSize}
    // 设置随机种子
    setRandomSeed(42);
        
    //若已有因子归因相关数据表，则不需通过如下代码模拟数据
    symbols = format(600001..601000, "000000") + ".SH"//1000支股票 
    dateTable   = table(startDate..endDate as date)
    symbolTable = table(symbols as symbol)
    baseTable   = cj(dateTable, symbolTable)  // 日期 × 股票 笛卡尔积
    n = rows(baseTable)   //n 支股票组合 
    
    // 1. 构造 bench（基准组合权重）
    benchTemp = table(baseTable, rand(1.0, n) as rawWeight)
    bench = select date, symbol, rawWeight / sum(rawWeight) as weight from benchTemp
            context by date order by date, symbol
    share(bench,"{self.benchPosObj}");
    
    // 2. 构造 market（个股收益）
    market = select date, symbol, randNormal(0.0, 0.03, n) as marketReturn from baseTable
             order by date, symbol
    share(market,"{self.returnObj}");
    
    // 3. 构造 own（投资组合权重）
    ownTemp = table(baseTable, rand(1.0, n) as rawWeight)
    own = select date, symbol, rawWeight / sum(rawWeight) as weight from ownTemp
          context by date order by date, symbol
    share(own,"{self.ownPosObj}");
    
    // 4. 构造 styleExpos（风格因子暴露）
    colList = {self.factor_list};
    q = size(colList);
    styleExpos = select date, symbol from baseTable
    for(i in 0..(q-1)){{
        vec = rand(2.0, n) - 1.0
         col = colList[i] //`style_+ string(i)
        styleExpos[col] = vec
    }}
    share(styleExpos,"{self.styleFactorObj}");
    
    //5. 构造 industryExpos（行业因子暴露）
    p = 30;
    industryColNames = "industry_" + string(1..p)  
    // industryColNames = `industry_电子`industry_汽车`industry_纺织服饰`industry_医药生物`industry_食品饮料`industry_非银金融`industry_有色金属`industry_商贸零售`industry_社会服务`industry_国防军工`industry_传媒`industry_机械设备`industry_银行`industry_计算机`industry_电力设备`industry_交通运输`industry_钢铁`industry_公用事业`industry_建筑装饰`industry_房地产`industry_基础化工`industry_石油石化`industry_通信`industry_煤炭`industry_家用电器`industry_美容护理`industry_环保`industry_建筑材料`industry_农林牧渔`industry_轻工制造`industry_综合
    // p = size(industryColNames);
    // 创建包含 date, symbol 和 industry_code 的基础查询  
    baseQuery = select date, symbol, (hashBucket(symbol, p) + 1) as industry_code from baseTable  
    industryExpos = select date, symbol from baseQuery  
      
    //循环生成并添加哑变量列  
    for(i in 1..p) {{
        colName = "industry_" + string(i)  
        industryExpos[colName] = iif(baseQuery.industry_code == i, 1, 0)  
    }}
    share(industryExpos,"{self.industryFactorObj}");
    """)