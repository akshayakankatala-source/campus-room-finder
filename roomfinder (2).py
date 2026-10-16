"""
Campus Room Finder - Round 1 base prototype
-------------------------------------------
Core idea: build ONE occupancy grid  occ[room, day, hour] = True/False
from classes (weekly, repeat every week) + events (specific dates).
Everything else (room finder, utilization, unmet-demand analysis)
is a query on that grid.
"""
import numpy as np
import pandas as pd

DAY0 = pd.Timestamp("2025-08-01")           # data window start
DAY1 = pd.Timestamp("2025-11-28")           # data window end
DAYS = pd.date_range(DAY0, DAY1)
WD = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
OPEN_H, CLOSE_H = 8, 20                     # campus "bookable day" 08:00-20:00


def load(path="."):
    rooms = pd.read_csv(f"{path}/rooms.csv")
    classes = pd.read_csv(f"{path}/class_schedule.csv")
    events = pd.read_csv(f"{path}/events.csv", parse_dates=["start_time"])
    reqs = pd.read_csv(f"{path}/room_requests.csv", parse_dates=["request_date"])
    return rooms, classes, events, reqs


def build_grid(rooms, classes, events):
    """occ[r, d, h] = True if room r is busy on day d (index into DAYS) at hour h."""
    ridx = {rid: i for i, rid in enumerate(rooms.room_id)}
    occ = np.zeros((len(rooms), len(DAYS), 24), dtype=bool)
    class_occ = np.zeros_like(occ)
    day_wd = np.array([d.weekday() for d in DAYS])

    for row in classes.itertuples():                       # weekly recurring
        r, w = ridx[row.room_id], WD[row.weekday]
        days = np.where(day_wd == w)[0]
        for h in range(row.start_hour, min(24, row.start_hour + row.duration_hours)):
            class_occ[r, days, h] = True

    ev_occ = np.zeros_like(occ)
    for row in events.itertuples():                        # one-off dated events
        r = ridx[row.room_id]
        start = row.start_time
        for k in range(row.duration_hours):
            t = start + pd.Timedelta(hours=k)
            d = (t.normalize() - DAY0).days
            if 0 <= d < len(DAYS):
                ev_occ[r, d, t.hour] = True
    return class_occ | ev_occ, class_occ, ev_occ, ridx


class RoomFinder:
    def __init__(self, path="."):
        self.rooms, self.classes, self.events, self.reqs = load(path)
        self.occ, self.class_occ, self.ev_occ, self.ridx = build_grid(
            self.rooms, self.classes, self.events)

    # ---------- availability ----------
    def free_mask(self, date, start_hour, duration):
        """Boolean per room: free for the whole [start, start+duration) window."""
        d = (pd.Timestamp(date) - DAY0).days
        hrs = range(start_hour, start_hour + duration)
        return ~self.occ[:, d, list(hrs)].any(axis=1)

    # ---------- room finder ----------
    def find_rooms(self, date, start_hour, duration, people,
                   projector=False, computers=False, accessible=False,
                   building=None, top_n=5):
        free = self.free_mask(date, start_hour, duration)
        r = self.rooms.copy()
        ok = free & (r.capacity >= people)
        if projector:  ok &= r.projector == 1
        if computers:  ok &= r.computers == 1
        if accessible: ok &= r.accessible == 1
        if building:   ok &= r.building == building
        c = r[ok].copy()
        if c.empty:
            return c
        c["fill_pct"] = (100 * people / c.capacity).round()
        # best fit = least wasted seats; small bonus for extras the user didn't need
        waste = (c.capacity - people) / c.capacity
        extras = ((c.projector == 1) & (not projector)).astype(int) * 0.02 \
               + ((c.computers == 1) & (not computers)).astype(int) * 0.02
        c["score"] = (1 - waste - extras).round(3)
        c = c.sort_values("score", ascending=False).head(top_n)
        end = start_hour + duration
        c["why"] = [self._explain(x, people, start_hour, end, projector, computers, accessible)
                    for x in c.itertuples()]
        return c[["room_id", "building", "floor", "room_type", "capacity",
                  "fill_pct", "score", "why"]].reset_index(drop=True)

    @staticmethod
    def _explain(x, people, s, e, proj, comp, acc):
        bits = [f"Capacity {x.capacity} fits {people} people ({x.fill_pct:.0f}% full)"]
        if proj: bits.append("projector available")
        if comp: bits.append("computers available")
        if acc:  bits.append("step-free accessible")
        bits.append(f"no class or event {s:02d}:00-{e:02d}:00")
        return ", ".join(bits)

    # ---------- smart fallback ----------
    def alternatives(self, date, start_hour, duration, people, **kw):
        """If nothing is free, look +/-1..3h on the same day."""
        out = []
        for shift in (1, -1, 2, -2, 3, -3):
            s = start_hour + shift
            if s < OPEN_H or s + duration > CLOSE_H:
                continue
            res = self.find_rooms(date, s, duration, people, top_n=1, **kw)
            if not res.empty:
                res.insert(0, "start_hour", s)
                out.append(res)
        return pd.concat(out).reset_index(drop=True) if out else pd.DataFrame()

    # ---------- utilization ----------
    def utilization(self):
        """% of bookable hours (Mon-Sat, 08-20) each room is occupied."""
        wd = np.array([d.weekday() for d in DAYS])
        day_sel = wd <= 5
        win = self.occ[:, day_sel, OPEN_H:CLOSE_H]
        u = win.mean(axis=(1, 2)) * 100
        cu = self.class_occ[:, day_sel, OPEN_H:CLOSE_H].mean(axis=(1, 2)) * 100
        eu = self.ev_occ[:, day_sel, OPEN_H:CLOSE_H].mean(axis=(1, 2)) * 100
        df = self.rooms.copy()
        df["util_pct"], df["class_pct"], df["event_pct"] = u.round(1), cu.round(1), eu.round(1)
        return df

    def heatmap(self):
        """campus-wide % of rooms busy by weekday x hour."""
        d = pd.DataFrame(index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                         columns=range(OPEN_H, CLOSE_H), dtype=float)
        wd = np.array([x.weekday() for x in DAYS])
        for i, name in enumerate(d.index):
            sel = wd == i
            for h in d.columns:
                d.loc[name, h] = self.occ[:, sel, h].mean() * 100
        return d.round(1)

    # ---------- demand vs supply ----------
    def request_analysis(self):
        """For each request: was a suitable room actually free? (the 'space exists' proof)"""
        out = []
        for q in self.reqs.itertuples():
            free = self.free_mask(q.request_date, q.start_hour, q.duration_hours)
            ok = free & (self.rooms.capacity.values >= q.required_capacity)
            if q.projector_required:  ok &= self.rooms.projector.values == 1
            if q.computers_required:  ok &= self.rooms.computers.values == 1
            out.append(int(ok.sum()))
        df = self.reqs.copy()
        df["suitable_free_rooms"] = out
        df["weekday"] = df.request_date.dt.day_name().str[:3]
        return df


if __name__ == "__main__":
    rf = RoomFinder(".")
    print("Rooms free for 45 people, projector, Tue 2025-09-09 14:00-16:00:")
    print(rf.find_rooms("2025-09-09", 14, 2, 45, projector=True).to_string())
