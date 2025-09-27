from src.FactorEva import SingleFactorBackTest

def get_DayLabel(self: SingleFactorBackTest, k: int, labelName: str):
    """
    统一的日频标签生成函数
    -> return table(
        ["symbol","tradeDate","tradeTime","method","label","maxDate","maxTime"]
    )
    """
    kdbName = "dfs://DayKDB"
    ktbName = "o_tushare_a_stock_daily"
    symbolCol = "ts_code"
    dateCol = "trade_date"
    openCol = "open"
    closeCol = "close"
    adjFactorCol = "adj_factor"
    # 定义标的池
    self.session.run(rf"""
    // idx_code = "000852.SH" // 1000
    // idx_code = "000905.SH" // 500
    // idx_code = "399300.SZ" // 300
    totalCodeList = array(STRING,0);
    for (idx_code in ["000852.SH", "000905.SH", "399300.SZ"]){{
        code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") 
            where index_code == idx_code and (trade_date between {self.start_dot_date} and {self.end_dot_date});
        totalCodeList.append!(code_list);
    }}
    """)

    # 定义数据插入函数
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

    self.session.run(f"""
       label_df = select {symbolCol} as symbol, 
                         {dateCol} as TradeDate,
                         {openCol}*{adjFactorCol} as open,
                         {closeCol}*{adjFactorCol} as close
                         from loadTable("{kdbName}", "{ktbName}")
                         where ({dateCol} between {self.start_dot_date} and temporalAdd({self.end_dot_date},{k},"B")) 
                         and {symbolCol} in totalCodeList
                         order by {dateCol},{symbolCol}
       label_df = select symbol, 
                         TradeDate, 
                         15:00:00.000 as TradeTime,
                         "{labelName}" as labelName,
                         nullFill((move(close,-{k})-open)\open,0) as label,
                         nullFill(move(TradeDate,-{k}).date(),temporalDeltas(TradeDate,"XSHG")) as MaxDate,
                         15:00:00.000 as MaxTime
                   from label_df 
                   context by symbol
       InsertData("{self.labelDB}", "{self.labelTB}", label_df, 1000000);
       undef(`label_df); // 释放内存
       """)

def get_MinLabel(self: SingleFactorBackTest, k: int, labelName: str):
    """
    统一的分钟频标签生成函数
    -> return table(
        ["symbol","tradeDate","tradeTime","method","label","maxDate","maxTime"]
    )
    """
    kdbName = "dfs://MinKDB"
    ktbName = "Min1K"
    symbolCol = "code"
    dateCol = "tradeDate"
    timeCol = "tradeTime"
    openCol = "open"
    closeCol = "close"

    self.session.run(f"""
       label_df = select {symbolCol} as symbol, 
                         {dateCol} as TradeDate,
                         {timeCol} as TradeTime,
                         {openCol} as open,
                         {closeCol} as close
                         from loadTable("{kdbName}", "{ktbName}")
                         where {dateCol} between {self.start_dot_date} and temporalAdd({self.end_dot_date},{int(k/240)},"B")
                         and {symbolCol} in totalCodeList
                         order by {dateCol}, {timeCol}, {symbolCol}
       label_df = select symbol, 
                         TradeDate, 
                         TradeTime,
                         "{labelName}" as labelName,
                         nullFill((move(close,-{k})-open)\open,0) as label,
                         nullFill(move(TradeDate,-{k}).date(),temporalDeltas(TradeDate,"XSHG")) as MaxDate,
                         nullFill(move(TradeTime,-{k}).time(),max(TradeTime)) as MaxTime
                   from label_df 
                   context by symbol
       InsertData("{self.labelDB}", "{self.labelTB}", label_df, 1000000);
       undef(`label_df); // 释放内存
       """)

def get_DayBarLabel(self: SingleFactorBackTest):
    get_DayLabel(self, 1, "dayBarReturn")

def get_DayLabel1(self: SingleFactorBackTest):
    get_DayLabel(self, 1, "ret1D")

def get_DayLabel5(self: SingleFactorBackTest):
    get_DayLabel(self, 5, "ret5D")

def get_DayLabel10(self: SingleFactorBackTest):
    get_DayLabel(self, 10, "ret10D")

def get_DayLabel20(self: SingleFactorBackTest):
    get_DayLabel(self, 20, "ret20D")

def get_MinBarLabel(self: SingleFactorBackTest):
    get_MinLabel(self, 1, "minBarReturn")

def get_MinLabel120(self: SingleFactorBackTest):
    get_MinLabel(self, 120, "ret120M")

def get_MinLabel240(self: SingleFactorBackTest):
    get_MinLabel(self, 240, "ret240M")

def get_MinLabel480(self: SingleFactorBackTest):
    get_MinLabel(self, 480, "ret480M")

def get_MinLabel1200(self: SingleFactorBackTest):
    get_MinLabel(self, 1200, "ret1200M")

