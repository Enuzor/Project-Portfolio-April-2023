#Genre-Based Analysis
#1 Total Number of Films Rented in each category
WITH Fav_CategoryCTE AS
(SELECT customer.customer_id, category.name AS "Category_Name", count( DISTINCT rental.rental_id) AS "Number_Category_rented"
FROM customer
JOIN rental USING(customer_id)
JOIN inventory USING(inventory_id)
JOIN film USING(film_id)
JOIN film_category USING(film_id)
JOIN category USING(category_id)
GROUP BY category.category_id, customer.customer_id , category.name
ORDER BY customer.customer_id, count(rental.rental_id)
) , Top_Fav_Category AS (SELECT * , DENSE_RANK() OVER(PARTITION BY customer_id ORDER BY Number_Category_Rented desc) AS Category_Rank
FROM Fav_CategoryCTE
)
Select Category_Name, count(Category_Name) AS "Count_of_Category_Rented"
FROM Top_Fav_Category TFC
WHERE Category_Rank = 1
GROUP BY Category_Name;

#2. The top 10 genres in terms of revenue & frequency

Select film_category.category_id,
category.name,sum(payment.amount) AS Revenue, 
count(inventory.film_id) as frequency
From film_category 
Inner Join inventory On film_category.film_id = inventory.film_id 
Inner Join rental On rental.inventory_id = inventory.inventory_id 
Inner Join payment on rental.rental_id=payment.rental_id
Inner Join category on film_category.category_id = category.category_id group by film_category.category_id
Order by Revenue desc;


#3 Top 10 genres in terms of frequency
JOIN film USING(film_id)
JOIN film_category USING(film_id)
JOIN category USING(category_id)
GROUP BY category.category_id, customer.customer_id , category.name
ORDER BY customer.customer_id, count(rental.rental_id)
) , Top_Fav_Category AS (SELECT * , DENSE_RANK() OVER(PARTITION BY customer_id ORDER BY Number_Category_Rented desc) AS Category_Rank
FROM Fav_CategoryCTE
)
Select Category_Name, count(Category_Name) AS "Count_of_Category_Rented"
FROM Top_Fav_Category TFC
WHERE Category_Rank = 1
GROUP BY Category_Name;

#based on frequency
Select film_category.category_id,
category.name,
count(inventory.film_id ) as frequency,
sum(payment.amount) AS Revenue
From film_category 
Inner Join inventory On film_category.film_id = inventory.film_id 
Inner Join rental On rental.inventory_id = inventory.inventory_id 
Inner Join payment on rental.rental_id=payment.rental_id
inner join category on film_category.category_id = category.category_id group by film_category.category_id 
order by frequency desc
limit 10;

#based on revenue
Select film_category.category_id,
category.name,sum(payment.amount) AS Revenue, 
count(inventory.film_id) as frequency
From film_category 
Inner Join inventory On film_category.film_id = inventory.film_id 
Inner Join rental On rental.inventory_id = inventory.inventory_id 
Inner Join payment on rental.rental_id=payment.rental_id
Inner Join category on film_category.category_id = category.category_id group by film_category.category_id
Order by Revenue desc
limit 10;

#3. The most popular genre in each store in terms of revenue and frequency
#based on frequency
#based on frequency
with c as (
Select store.store_id,
film_category.category_id,
category.name, 
count(inventory.film_id) as frequency,
sum(payment.amount) revenue
From store
Inner Join inventory On inventory.store_id = store.store_id
Inner Join rental On rental.inventory_id = inventory.inventory_id 
Inner Join payment On payment.rental_id = rental.rental_id
Inner Join film_category On film_category.film_id = inventory.film_id Inner Join category On film_category.category_id = category.category_id
group by store.store_id,film_category.category_id
order by frequency desc
)
select c.store_id, c.name, max(c.frequency) frequency,c.revenue from c
group by c.store_id;

#based on revenue
with c as (
Select store.store_id,
film_category.category_id, 
category.name,sum(payment.amount) Revenue ,
count(inventory.film_id) as frequency
From store
Inner Join inventory On inventory.store_id = store.store_id
Inner Join rental On rental.inventory_id = inventory.inventory_id 
Inner Join payment On payment.rental_id = rental.rental_id
Inner Join film_category On film_category.film_id = inventory.film_id Inner Join category On film_category.category_id = category.category_id
group by store.store_id,film_category.category_id
order by Revenue desc
)
select c.store_id, c.category_id, c.name, max(c.Revenue), c.frequency 
from c 
group by c.store_id;

#Film-Based Analysis
#The top 20 films in each category in terms of frequency and revenue
#based on revenue
SET @userinput1 :=14;
Select film_category.category_id,category.name,film_category.film_id, sum(payment.amount) revenue,count(inventory.film_id) as frequency
From film_category
Inner Join inventory On film_category.film_id = inventory.film_id
Inner Join rental On rental.inventory_id = inventory.inventory_id
Inner Join payment On payment.rental_id = rental.rental_id
Inner Join category On film_category.category_id = category.category_id
WHERE film_category.category_id = @userinput1
group by film_category.film_id, film_category.category_id
order by film_category.category_id Asc , revenue Desc
limit 20;

#The top 10 films for each rating (rating: PG, PG14 etc)
Select
film.film_id,
film.title,
film.rating,sum(payment.amount)AS revenue,
count(inventory.film_id) AS frequency
From film
Inner Join inventory On inventory.film_id = film.film_id Inner Join rental On rental.inventory_id = inventory.inventory_id
Inner Join payment On payment.rental_id = rental.rental_id
where film.rating
group by film.film_id
order by revenue Desc
limit 10;

#3. The top 10 movies in terms of language (revenue and frequency)
#based on frequency
SET @userinput1 :=1;
Select language.language_id,language.name,film.film_id, film.title, count(inventory.film_id) as frequency,sum(payment.amount) revenue
From `language`
Inner Join film On film.language_id = language.language_id Inner Join inventory On inventory.film_id = film.film_id 
Inner Join rental On rental.inventory_id = inventory.inventory_id
Inner Join payment On payment.rental_id = rental.rental_id where language.language_id = @userinput1
group by film.film_id
order by frequency Desc
limit 10;

#based on revenue
Select language.language_id, language.name,film.film_id, film.title, sum(payment.amount) revenue,count(inventory.film_id) as frequency
From language
Inner Join film On film.language_id = language.language_id 
Inner Join inventory On inventory.film_id = film.film_id
Inner Join rental On rental.inventory_id = inventory.inventory_id
Inner Join payment On payment.rental_id = rental.rental_id
where language.language_id
group by film.film_id
order by revenue Desc
limit 10;

#Number of movies per actor
with abc as (SELECT distinct(actor_id) actor_id,count(film_id) film_perform
FROM film_actor fa
group by actor_id
order by actor_id asc)
select abc.actor_id,a.first_name,a.last_name, abc.film_perform from abc inner join actor a on abc.actor_id=a.actor_id order by abc.film_perform desc
limit 10;

#Store-Based Analysis
#1How many rented films were returned on time, late, or early? (DVD Return Rate)

WITH t1 as (
select rental_id, return_date, rental_date, f.rental_duration,
case when(rental_duration > datediff(return_date , rental_date)) THEN "Returned early"
when (rental_duration = datediff(return_date , rental_date)) THEN "Returned on time"
else "Returned late"
End AS return_status
from rental
join inventory inv using (inventory_id)
join film f on inv.film_id = f.film_id
)
select return_status , count(*) AS "Total_films" -- add percentage 
from t1
group by return_status;

#2. Revenue generated by each store.

#Store Monthly Sales
WITH Percentage_Wise_Monthly_Revenue AS(
SELECT s.store_id, YEAR(p.payment_date) AS 'Year' , MONTHNAME(p.payment_date) AS 'Month' , SUM(p.amount) AS Total_Sales_Monthly FROM store s
JOIN staff st USING(store_id)
JOIN payment p USING(staff_id)
GROUP BY store_id, Year, Month
)
SELECT * , Total_Sales_Monthly*100/(SELECT SUM(Total_Sales_Monthly) FROM Percentage_Wise_Monthly_Revenue WHERE store_id=PWMR.store_id) AS "Sales_Per_Store_Monthly_Percent"
FROM Percentage_Wise_Monthly_Revenue PWMR ORDER BY store_id, PWMR.Year, PWMR.Month desc;
#Revenue generated by each store
WITH Percentage_Wise_Revenue AS(
SELECT store_id, sum(amount) AS "Total_Sales", count(rental_id) AS "Total_films_rented"
FROM store s
JOIN staff USING(store_id)
JOIN payment USING(staff_id)
GROUP BY store_id
ORDER BY store_id
)
SELECT * , Total_Sales*100/(SELECT SUM(Total_Sales) FROM Percentage_Wise_Revenue) AS "Sales_Per_Store_Percent"
FROM Percentage_Wise_Revenue
GROUP BY store_id;

#Country-Based Analysis
#Revenue generated by each country
#Top 10 Cuntries
config()->set('database.connections.mysql.strict', false);

select cs.first_name AS Name, 
cs.email AS Email , 
ad.address, 
ad.district, 
c.country,
ci.city, 
sum(p.amount) AS Sales
from country c
join city ci using (country_id)
join address ad on ci.city_id = ad.city_id
join customer cs on cs.address_id = ad.address_id
join payment p on cs.customer_id = p.customer_id
group by country
order by Sales desc
limit 10;

Select
country.country_id,country.country, count(distinct(city.city_id)) as city_count , count(distinct(customer.customer_id)) as customer_count ,count(rental.rental_id) as frequency, sum(payment.amount) as Revenue
From customer
Inner Join address On customer.address_id = address.address_id
Inner Join city On address.city_id = city.city_id Inner Join country on city.country_id = country.country_id Inner Join rental On rental.customer_id = customer.customer_id
Inner Join payment On payment.customer_id = customer.customer_id And payment.rental_id = rental.rental_id
group by country.country_id;


#Recommendations
SET @userid :='1';
with top_category as (
with ranking as(
Select c.customer_id,cate.category_id,
count(i.film_id) as frequency, sum(p.amount) revenue 
From customer c
Inner Join payment p On p.customer_id = c.customer_id Inner Join rental r On r.customer_id = c.customer_id
And p.rental_id = r.rental_id
Inner Join inventory i On r.inventory_id = i.inventory_id Inner Join film f On i.film_id = f.film_id
Inner Join film_category fc On fc.film_id = f.film_id
Inner Join category cate On fc.category_id = cate.category_id where c.customer_id = @userid
group by cate.category_id,c.customer_id 
order by Revenue desc)
select ranking.customer_id,ranking.category_id,ranking.frequency,ranking.revenue, DENSE_RANK() OVER (PARTITION BY ranking.customer_id ORDER BY frequency
DESC)frequency_rank
FROM ranking
)
Select fc.category_id,cate.name,fc.film_id,f.title, count(i.film_id) as frequency,sum(p.amount) revenue 
From film_category fc
Inner Join inventory i On fc.film_id = i.film_id
Inner Join rental r On r.inventory_id = i.inventory_id
Inner Join payment p On p.rental_id = r.rental_id
Inner Join category cate On fc.category_id = cate.category_id 
Inner Join film f On i.film_id = f.film_id
where fc.category_id =(select top_category.category_id from top_category
where frequency_rank=1 limit 1) AND fc.film_id NOT IN (
Select distinct(i.film_id) as watched
From inventory i
Inner Join rental r On r.inventory_id = i.inventory_id
Inner Join customer c On r.customer_id = c.customer_id
where c.customer_id=@userid
order by watched
)
group by fc.film_id, fc.category_id
order by fc.category_id Asc , frequency Desc,revenue desc 
limit 10;

SELECT i.film_id, f.title, COUNT(i.film_id) AS total_number_of_rental_times, f.rental_rate, COUNT(i.film_id)*f.rental_rate AS revenue_per_movie
FROM rental r
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN film f ON f.film_id = i.film_id
GROUP BY i.film_id
ORDER BY 5 DESC;

SELECT left(rental_date,7) AS "Month", COUNT(*)
FROM rental
GROUP BY 1;

/* Reward Users : who has rented at least 30 times*/
DROP TEMPORARY TABLE IF EXISTS tbl_rewards_user;
CREATE TEMPORARY TABLE tbl_rewards_user(
SELECT r.customer_id, COUNT(r.customer_id) AS total_rents, max(r.rental_date) AS last_rental_date
FROM rental r
GROUP BY 1
HAVING COUNT(r.customer_id) >= 30);
/* Reward Users who are also active */
SELECT au.customer_id, au.first_name, au.last_name, au.email
FROM tbl_rewards_user ru
JOIN tbl_active_users au ON au.customer_id = ru.customer_id;

/* All Rewards Users with Phone */
SELECT ru.customer_id, c.email, au.phone
FROM tbl_rewards_user ru
LEFT JOIN tbl_active_users au ON au.customer_id = ru.customer_id
JOIN customer c ON c.customer_id = ru.customer_id;

#How many distint Renters per month*/
SELECT LEFT(rental_date,7) AS "Month", 
	COUNT(DISTINCT(rental_id)) AS "Total Rentals",
	COUNT(DISTINCT(customer_id)) AS "Number Of Unique Renter", 
    COUNT(DISTINCT(rental_id))/COUNT(DISTINCT(customer_id)) AS "Average Rent Per Renter"
FROM rental
GROUP BY 1;
