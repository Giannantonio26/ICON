
        (strong_dribbler(Player) :- player_stats(_, Player, _, _, _, Min, ToSuc, _, _, _, _, _, _, _), ToSuc > 1.10, Min > 1000).
        (strong_playmaker(Player) :- player_stats(_, Player, _, _, _, Min, _, Rec, RecProg, PasTotCmp, PasAss, PasCmp, PasProg, PasLonCmp),  Min>1000, Rec>34, RecProg>3, PasTotCmp>33, PasAss>0.85, PasCmp>33, PasProg>3, PasLonCmp>3).
    
        (strong_dribbler(Player) :- player_stats(_, Player, _, _, _, Min, ToSuc, _, _, _, _, _, _, _), ToSuc > 1.10, Min > 1000).
        (strong_playmaker(Player) :- player_stats(_, Player, _, _, _, Min, _, Rec, RecProg, PasTotCmp, PasAss, PasCmp, PasProg, PasLonCmp),  Min>1000, Rec>34, RecProg>3, PasTotCmp>33, PasAss>0.85, PasCmp>33, PasProg>3, PasLonCmp>3).
    
        (strong_dribbler(Player) :- player_stats(_, Player, _, _, _, Min, ToSuc, _, _, _, _, _, _, _), ToSuc > 1.10, Min > 1000).
        (strong_playmaker(Player) :- player_stats(_, Player, _, _, _, Min, _, Rec, RecProg, PasTotCmp, PasAss, PasCmp, PasProg, PasLonCmp),  Min>1000, Rec>34, RecProg>3, PasTotCmp>33, PasAss>0.85, PasCmp>33, PasProg>3, PasLonCmp>3).
    