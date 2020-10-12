This project has been done for educational purposes only and not for commercial purposes.

https://www.gov.uk/guidance/exceptions-to-copyright

## AUC Score of ~0.87 on validation data

## Why this was done (more detail in the notebook)

I wanted to work on a problem that could grow my data skills and on something that (to my knowledge) has little or no one focusing on it. 

Brazilian Jiu Jitsu 🥋 is a grappling sport that is still developing on the mat but not in the data world so I developed a baseline model for predicting matches. 

As mentioned in the notebook I see there being a lot of different ways you can use data that isn't being used.

## Who this benefits (both data and modelling)
* **Match makers** - pair competitors with similar performances, certain styles or close matches (equal probabilities of winning)
* **Coaches** - use data to analyse who their athlete's next competitor is and their style both currently and historically to improve training approaches for competition
* **Analysts/commentators** - stats about a fighter can be useful but when you have to watch hours of footage to get them, not efficient. The simple features added to the data can help find stats about a fighter to date or historically. Also some stats that aren't necessarily obvious or easy to compute using a brain could be encorporated into live commentatary. 

## Why I haven't included the data collection code
I wasn't happy with the way I collected and manipulated the data especially towards the end of the project as I was keen to get modelling ASAP so the code was fairly messy in the final stages hence I have chosen to omit it.

As mentioned in the notebook the majority of features were cumulatively summed and lagged to create features "to date".

Most of the code was a function (in R) similar to this

```points_tracking <- function (matches) {
 matches %>%                                                             # Feed matches in
    arrange(fighter_a, id) %>%                                           # Arrange by fighter and earliest to latest fight
    group_by(fighter_a, year) %>%                                        # Group by each fighter and year
    mutate(win = if_else(w_l == "w" & method_cat == "points", 1, 0),     # Create cumulative wins and losses via points
           lose = if_else(w_l == "l" & method_cat == "points", 1, 0),
           points_win_TD = cumsum(win),
           points_lose_TD = cumsum(lose),
           cum_win_points = lag(points_win_TD, default = 0),
           cum_lose_points = lag(points_lose_TD, default = 0)) %>%
    select(-win, -lose, -ends_with("TD")) %>%                            # Remove unwanted columns
    ungroup()                                                            # Ungroup
}
```
