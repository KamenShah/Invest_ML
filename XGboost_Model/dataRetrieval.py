import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import time
import datetime
from datetime import date
from datetime import timedelta
import csv
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pickle
import gzip



header = "Stock,Date,EPS Reported,EPS Surprise,Beat Estimate,EPS Estimate,Net Profit Margin,Return On Assets,Current EPS Growth,Fiscal Year EPS Growth,Cash Flow Accruals,Balance Sheet Accruals,Quick Ratio,Price Earnings Ratio,Debt Equity Ratio,Return On Equity,Current Ratio,Price To Sales Ratio,Price To Book Ratio,Dividend Yield,Dividend Payout Ratio,Asset Turnover,Inventory Turnover,Cash Ratio,Gross Profit Growth,EPS Growth,Free Cash Flow Growth,Receivables Growth,Asset Growth,Debt Growth,R&D Growth\n"

def constructURL(date):
    today = date.today()
    url = "https://finance.yahoo.com/calendar/earnings"
    if(date == today):
        return url
    else:
        month = str(date.month)
        year = str(date.year)
        day = str(date.day)
        if(date.month < 10):
            month = "0" + month
        if(date.day < 10):
            day = "0" + day
        url = url + "?day=" + year + "-" + month + "-" + day
        return url
    
def getStocks(url, date, test=False):   
    stocks = []
    try:
        response = requests.get(url)
    except:
        raise Exception("Error occured in calling request at",date)
    soup = BeautifulSoup(response.text, "html.parser")
    objects = soup.findAll("a", class_="Fw(600)")
    for i in range(0, len(objects)):
        estimate = None
        reported = None
        surprise = None
        stock = objects[i].contents[0]
        beat = None
        colunms = objects[i].findParent().findNextSiblings()
        for i in colunms:
            if(i.text == 'N/A'):
                continue
            elif( i['aria-label'] == "EPS Estimate"):
                estimate = i.text
            elif( i['aria-label'] == "Reported EPS"):
                reported = i.text
            elif( i['aria-label'] == "Surprise(%)"):
                surprise = i.text
        if(surprise != None):
            if(float(surprise) > 0):
                beat = 1
            else:
                beat = -1
        if(surprise == None and test == False):
            continue
        stocks.append({"stock": stock, 'date': date, 'epsReported': reported, 'epsSurprise': surprise, 'beatEstimate': beat, 'epsEstimate': estimate})
    return stocks



#https://github.com/antoinevulcain/Financial-Modeling-Prep-API
def getData(stock, date):
    annualIncomeStatementURL = "https://financialmodelingprep.com/api/v3/financials/income-statement/"+stock
    quarterlyIncomeStatementURL = "https://financialmodelingprep.com/api/v3/financials/income-statement/"+stock+"?period=quarter"
    annualBalanceSheetURL = "https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/"+stock
    quarterlyBalanceSheetURL =  "https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/"+stock+"?period=quarter"
    annualCashFlowURL = "https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/"+stock
    quarterlyCashFlowURL = "https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/"+stock+"?period=quarter"
    annualFinancialGrowthURL = "https://financialmodelingprep.com/api/v3/financial-statement-growth/"+stock
    quarterlyFinancialGrowthURL = "https://financialmodelingprep.com/api/v3/financial-statement-growth/"+stock+"?period=quarter"
    ratingURL = "https://financialmodelingprep.com/api/v3/company/rating/"+stock
    ratiosURL = "https://financialmodelingprep.com/api/v3/financial-ratios/"+stock
    quaterlyGrowthURL = "https://financialmodelingprep.com/api/v3/financial-statement-growth/"+stock+"?period=quarter"
    

    try:
        QIncomeResponse = requests.get(quarterlyIncomeStatementURL)
        QBalanceResponse = requests.get(quarterlyBalanceSheetURL)
        ABalanceResponse = requests.get(annualBalanceSheetURL)
        AFinancialGrowth = requests.get(annualFinancialGrowthURL)
        QFinancialGrowth = requests.get(quarterlyFinancialGrowthURL)
        QCashFlow = requests.get(quarterlyCashFlowURL)
        Ratios = requests.get(ratiosURL)
        QGrowth = requests.get(quaterlyGrowthURL)
        
        if(len(QIncomeResponse.json()) != 0 and len(QBalanceResponse.json()) != 0 and len(ABalanceResponse.json()) != 0 and len(AFinancialGrowth.json()) != 0 and len(QFinancialGrowth.json()) != 0 and len(QCashFlow.json()) != 0 and len(Ratios.json()) != 0 and len(QGrowth.json()) != 0 ):

            listings = QBalanceResponse.json()["financials"]
            listingQNum = 0
            for i in range(0, len(listings)):
                dateVal = listings[i]["date"].split("-")
                financialDate = datetime.date(int(dateVal[0]), int(dateVal[1].lstrip('0')), int(dateVal[2].lstrip('0'))) 
                if(financialDate < date):
                    break
                elif(i == len(listings)):
                    return None
                listingQNum += 1


            listings = ABalanceResponse.json()["financials"]
            listingANum = 0
            for i in range(0, len(listings)):
                dateVal = listings[i]["date"].split("-")
                financialDate = datetime.date(int(dateVal[0]), int(dateVal[1].lstrip('0')), int(dateVal[2].lstrip('0'))) 
                if(financialDate.year <= date.year):
                    break
                elif(i == len(listings)):
                    return None
                listingANum += 1

            netProfitMargin = convertToFloat(QIncomeResponse.json()["financials"][listingQNum]["Net Profit Margin"])
            netIncome = convertToFloat(QIncomeResponse.json()["financials"][listingQNum]["Net Income"])
            totalAssets = convertToFloat(QBalanceResponse.json()["financials"][listingQNum]["Total assets"])
            returnOnAssets = convertToFloat(netIncome)/ float(totalAssets)
            currentEPSGrowth = convertToFloat(QFinancialGrowth.json()["growth"][listingQNum]["EPS Diluted Growth"])
            fiscalYearEPSGrowth = convertToFloat(AFinancialGrowth.json()["growth"][listingANum]["EPS Diluted Growth"])
            operatingCashFlow = convertToFloat(QCashFlow.json()["financials"][listingQNum]["Operating Cash Flow"])
            cashFlowAccruals = (netIncome -operatingCashFlow) / totalAssets
            totalAssetsThisYear = convertToFloat(ABalanceResponse.json()["financials"][listingANum]["Total assets"])
            totalAssetsLastYear = convertToFloat(ABalanceResponse.json()["financials"][listingANum+1]["Total assets"])
            balanceSheetAccruals = (totalAssetsThisYear - totalAssetsLastYear)/(totalAssetsThisYear + totalAssetsLastYear)
            QuickRatio = convertToFloat(Ratios.json()["ratios"][listingANum]["liquidityMeasurementRatios"]["quickRatio"])
            priceEarningsRatio = convertToFloat(Ratios.json()["ratios"][listingANum]["investmentValuationRatios"]["priceEarningsRatio"])
            debtEquityRatio = convertToFloat(Ratios.json()["ratios"][listingANum]["debtRatios"]["debtEquityRatio"])
            returnOnEquity = convertToFloat(Ratios.json()["ratios"][listingANum]["profitabilityIndicatorRatios"]["returnOnEquity"])
            currentRatio =  convertToFloat(Ratios.json()["ratios"][listingANum]["liquidityMeasurementRatios"]["currentRatio"])
            priceToSalesRatio = convertToFloat(Ratios.json()["ratios"][listingANum]["investmentValuationRatios"]["priceToSalesRatio"]) 
            priceToBookRatio = convertToFloat(Ratios.json()["ratios"][listingANum]["investmentValuationRatios"]["priceToBookRatio"])
            dividendYield = convertToFloat(Ratios.json()["ratios"][listingANum]["investmentValuationRatios"]["dividendYield"])
            dividendPayoutRatio = convertToFloat(Ratios.json()["ratios"][listingANum]["cashFlowIndicatorRatios"]["dividendPayoutRatio"])
            assetTurnover =  convertToFloat(Ratios.json()["ratios"][listingANum]["operatingPerformanceRatios"]["assetTurnover"])
            inventoryTurnover = convertToFloat(Ratios.json()["ratios"][listingANum]["operatingPerformanceRatios"]["inventoryTurnover"])
            cashRatio = convertToFloat(Ratios.json()["ratios"][listingANum]["liquidityMeasurementRatios"]["cashRatio"])
            GrossProfitGrowth = convertToFloat(QGrowth.json()["growth"][listingQNum]["Gross Profit Growth"])
            EPSGrowth = convertToFloat(QGrowth.json()["growth"][listingQNum]["EPS Growth"])
            freeCashFlowgrowth = convertToFloat(QGrowth.json()["growth"][listingQNum]["Free Cash Flow growth"])
            receivablesGrowth = convertToFloat(QGrowth.json()["growth"][listingQNum]["Receivables growth"])
            assetGrowth = convertToFloat(QGrowth.json()["growth"][listingQNum]["Asset Growth"])
            debtGrowth = convertToFloat(QGrowth.json()["growth"][listingQNum]["Debt Growth"])
            RDGrowth =  convertToFloat(QGrowth.json()["growth"][listingQNum]["R&D Expense Growth"])
            return {"netProfitMargin": netProfitMargin, 'returnOnAssets': returnOnAssets,
                    'currentEPSGrowth': currentEPSGrowth, 'fiscalYearEPSGrowth': fiscalYearEPSGrowth,
                    'cashFlowAccruals': cashFlowAccruals, 'balanceSheetAccruals': balanceSheetAccruals,
                    'quickRatio': QuickRatio, 'priceEarningsRatio':priceEarningsRatio, 'debtEquityRatio':debtEquityRatio,
                    'returnOnEquity': returnOnEquity, 'currentRatio': currentRatio, 'priceToSalesRatio': priceToSalesRatio,
                    'priceToBookRatio': priceToBookRatio, 'dividendYield': dividendYield, 'dividendPayoutRatio': dividendPayoutRatio,
                    'assetTurnover': assetTurnover, 'inventoryTurnover': inventoryTurnover, 'cashRatio':cashRatio,
                    'GrossProfitGrowth': GrossProfitGrowth, 'EPSGrowth':EPSGrowth, 'freeCashFlowgrowth':freeCashFlowgrowth,
                    'receivablesGrowth': receivablesGrowth, 'assetGrowth':assetGrowth, 'debtGrowth':debtGrowth,
                    'RDGrowth': RDGrowth}
        else:
            print("ERROR no financial data for", stock)
            return None
    except Exception as e: 
        print("ERROR",e ,"occured at requesting", stock,"at",date)
        return None
    
    
def convertToFloat(x):
    try:
        return float(x)
    except:
        return 0.0

def createTrainingDataFile(startDate, endDate, fileName):
    todayDate = date.today()
    writeType = 'a'
    if(startDate > todayDate or endDate > todayDate or startDate > endDate):
        print("Error with Dates, make sure dates are before today and start date is before end date")
        return
    else:
        with open(fileName,writeType) as outputFile:
            if(len(list(csv.reader(open(fileName)))) == 0 or writeType == "w"):
                outputFile.write(header)
            currentDate = endDate
            while startDate <= currentDate:
                print("Reading stocks for",currentDate)
                URL = constructURL(currentDate)
                stocks = getStocks(URL, currentDate)
                for stock in stocks:
                    if(stock["epsSurprise"] == "N/A"):
                        print("NO EPS Surprise data for", stock["stock"])
                        continue
                    data = getData(stock["stock"], currentDate)
                    if(data == None):
                        continue
                    else:
                        row = ""
                        row += ','.join('{}'.format(val) for val in stock.values())
                        row += ','
                        row += ','.join('{}'.format(val) for val in data.values())
                        row += "\n"
                        outputFile.write(row)
                        print("Writing to output file for", stock["stock"])
                currentDate =  currentDate - timedelta(days=1)

                
            


def predictEarnings(date, model):
    URL = constructURL(date)
    stocks = getStocks(URL, date, True)
    df = pd.DataFrame( columns = header.rstrip().split(','))
    for stock in stocks:
        if(stock["epsEstimate"] == None):
            continue
        data = getData(stock["stock"], date)
        if(data == None):
            continue
        else:
            row = []
            for val in stock.values():
                row.append(val) 
            for val in data.values():
                row.append(val)     
            df.loc[len(df)] = row
    if(len(df) == 0):
        return "No results"

    df = df.replace('None', 0)
    df = df.loc[(df!='Stock').all(1)]
    df = df.dropna(axis=1,how='all')
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
#    df = df[(df != 0).all(1)]
    

    df = df.drop('Debt Equity Ratio', 1)
    df = df.drop('Dividend Yield', 1)
    df = df.drop('Dividend Payout Ratio', 1)
    df = df.drop('Inventory Turnover', 1)
    df = df.drop('Receivables Growth', 1)
    df = df.drop('Debt Growth', 1)
    df = df.drop('R&D Growth', 1)

    predictions = model.predict(df.loc[:, 'EPS Estimate':'Asset Growth'].values)
    retValues = {}
    for i,x in enumerate(df.loc[:, "Stock"]):
        if predictions[i] == 1 and float(df.loc[i,'EPS Estimate']) > 0:
            retValues[x] = "BEAT estimates"
        else:
            retValues[x] = "NOT BEAT estimates"
    return retValues


def createFileTest():
    startDate = datetime.date(2018, 9, 18) 
    endDate = datetime.date(2018, 9, 19)
    createTrainingDataFile(startDate, endDate, 'stockData2.csv')
   

