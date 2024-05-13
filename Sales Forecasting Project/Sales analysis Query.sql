--Show all customer's records
SELECT * 
FROM PortfolioProject.dbo.customers$

--Show total number of customers

SELECT count(*)
FROM PortfolioProject.dbo.customers$;

--Show transactions for Chenna market (market code for chennai is Mark001)

SELECT *
FROM PortfolioProject.dbo.transactions$
WHERE market_code='Mark001';

--Show distrinct product codes that were sold in chennai.

SELECT distinct product_code
FROM PortfolioProject.dbo.transactions$
where market_code='Mark001';

--Show transactions where currency is US dollars.

SELECT *
from PortfolioProject.dbo.transactions$
where PortfolioProject.dbo.transactions$.currency = "USD";

--Show transactions in 2020 join by date table.

SELECT *
FROM portfolioproject.dbo.transactions$ t
INNER JOIN portfolioproject.dbo.date$ d
ON t.order_date = d.date
where d.year = 2020

SELECt *
FROM PortfolioProject.dbo.transactions$

--Show total revenue in year 2020.

SELECT sum(t.sales_amount)
FROM PortfolioProject.dbo.transactions$ t
INNER JOIN PortfolioProject.dbo.date$ d
ON t.order_date = d.date

--Show total revenue in year 2020, January Month.

SELECT SUM(t.sales_amount)
FROM PortfolioProject.dbo.transactions$ t
INNER JOIN PortfolioProject.dbo.date$ d
ON t.order_date =d.date
where d.year=2020 and d.month_name="January";

--Show total revenue in year 2020 in Chennai.

SELECT *
FROM  PortfolioProject.dbo.transactions$ t
INNER JOIN PortfolioProject.dbo.date$ d
ON t.order_date = d.date
Where t.market_code = Mark001;

