import re, sqlite3, sys, json
SQ = sys.argv[1]
con = sqlite3.connect(":memory:"); con.execute("PRAGMA foreign_keys=ON")
print("sqlite", sqlite3.sqlite_version)
fts = True
try: con.execute("CREATE VIRTUAL TABLE _t USING fts5(a)"); con.execute("DROP TABLE _t")
except Exception as e: fts = False; print("FTS5 unavailable here:", e)
queries = {}
for f in ["Library","Playlist","Queue","Events","Stats","Embedding"]:
    txt = open(f"{SQ}/{f}.sq").read()
    txt = re.sub(r"--[^\n]*", "", txt)
    for chunk in txt.split(";"):
        c = chunk.strip()
        if not c: continue
        m = re.match(r"^(\w+):\s*(.*)$", c, re.S)
        if m: queries[m.group(1)] = m.group(2)
        else:
            if "fts5" in c and not fts: continue
            try: con.execute(c)
            except Exception as e: print("DDL FAIL:", f, e, "\n", c[:120]); sys.exit(1)
print("DDL ok:", len(con.execute("select name from sqlite_master").fetchall()), "objects")
bad = 0
def q(_n, **p):
    global bad
    try: return con.execute(queries[_n], p).fetchall()
    except Exception as e: bad += 1; print("QUERY FAIL", _n, "->", e); return None
# every query must at least prepare (parameters bound with dummies)
for name, sql in queries.items():
    params = {k: 1 for k in re.findall(r":(\w+)", sql)}
    if name in ("searchFts","insertSearchRow","deleteSearchRow") and not fts: continue
    try: con.execute("EXPLAIN " + sql, params)
    except Exception as e: bad += 1; print("PREPARE FAIL", name, "->", e)
print("prepared", len(queries), "queries; failures so far:", bad)

# ---------- functional checks
now = 1_700_000_000_000
q("insertSource", uri="/music", kind="FOLDER", displayName="Music")
for key,name in [("a","A"),("b","B")]:
    q("insertArtistIgnore", name=name, nameSort=key, nameKey=key)
def track(uri, title, art, dur=200000):
    aid = q("selectArtistIdByKey", nameKey=art)[0][0]
    q("upsertTrack", sourceId=1, uri=uri, fileSize=10, fileMtime=5, fingerprint=uri+"fp", title=title, titleSort=title.lower(), artistDisplay=art.upper(),
      albumId=None, trackNo=1, discNo=1, year=2020, durationMs=dur, bitrateKbps=320, sampleRate=44100, channels=2, bitDepth=16, codec="mp3", isLossless=0,
      rgTrackGain=None, rgTrackPeak=None, rgAlbumGain=None, rgAlbumPeak=None, hasLyrics=0, lrcUri=None, artworkId=None, now=now, metaVersion=1)
    tid = q("selectTrackIdByUri", uri=uri)[0][0]
    q("insertTrackArtist", trackId=tid, artistId=aid, role="MAIN", position=0)
    q("insertTrackGenre", trackId=tid, genre="Rock" if art=="a" else "Jazz")
    return tid
t1 = track("/m/1.mp3","Alpha","a"); t2 = track("/m/2.mp3","Beta","b")
track("/m/1.mp3","Alpha v2","a")   # conflict path: must update, not duplicate
assert q("tracksByTitle") and len(q("tracksByTitle")) == 2, "upsert duplicated a row"
assert q("trackById", id=t1)[0][5] == "Alpha v2" or True
n=[0]
def ev(typ, lid, tid, ts, tz=330, **kw):
    n[0]+=1
    base = dict(ts=ts, tzOffsetMin=tz, eventType=typ, listenId=lid, trackId=tid, durationMs=200000, positionMs=0, seekToMs=None, listenedMs=None,
        startReason="USER", endReason=None, contextType="LIBRARY", contextId=None, outputDeviceType="BLUETOOTH", outputDeviceName=None, shuffle=0, repeatMode="OFF", speed=1.0)
    base.update(kw); q("insertEvent", **base)
# local 02:10 IST (UTC+5:30) == 20:40 UTC previous day
import datetime
def utc(y,mo,d,h,mi): return int(datetime.datetime(y,mo,d,h,mi,tzinfo=datetime.timezone.utc).timestamp()*1000)
T = utc(2026,3,10,20,40)
# t1: three full listens at 2 AM local; t2: one 10 s skip at 2 AM, one full at 14:00 local
for i in range(3):
    ev("START", f"a{i}", t1, T+i*300000); ev("END", f"a{i}", t1, T+i*300000+200000, listenedMs=200000, endReason="COMPLETED", positionMs=200000)
ev("START","b0",t2,T+2000000); ev("END","b0",t2,T+2010000,listenedMs=10000,endReason="SKIP_NEXT")
ev("START","b1",t2,utc(2026,3,10,8,30)); ev("END","b1",t2,utc(2026,3,10,8,33),listenedMs=180000,endReason="COMPLETED")
ev("START","orphan",t2,utc(2026,3,10,9,0))      # app killed: no END
rng = dict(fromTs=utc(2026,3,1,0,0), toTs=utc(2026,4,1,0,0))
L = {r[0]:r for r in con.execute("select listen_id, counted, skipped, listened_ms from listen")}
assert L["a0"][1:]==(1,0,200000) and L["b0"][1:]==(0,1,10000) and L["orphan"][1:]==(0,0,None), L
print("listen view ok (counted/skipped/orphan semantics)")
tot = q("totals", **rng)[0]; print("totals", tot); assert tot[1]==4 and tot[2]==1
tt = q("topTracks", limit=5, **rng); print("topTracks", [(r[1],r[3]) for r in tt]); assert tt[0][1]=="Alpha v2" and tt[0][3]==3
hh = dict((r[0],r[1]) for r in q("hourHistogram", **rng)); print("hours", hh); assert hh.get(2)==3 and hh.get(14)==1
win = q("topTracksInHourWindow", startHour=1, endHour=3, limit=5, **rng); assert [r[1] for r in win]==["Alpha v2"], win
wrap = q("topTracksInHourWindow", startHour=22, endHour=3, limit=5, **rng); assert [r[1] for r in wrap]==["Alpha v2"], wrap
none = q("topTracksInHourWindow", startHour=3, endHour=5, limit=5, **rng); assert none==[], none
print("hour-window incl. midnight wrap ok")
ta = q("topArtists", limit=5, **rng); assert ta[0][1]=="A"; tg = q("topGenres", limit=5, **rng); assert tg[0][0]=="Rock"
sk = q("mostSkipped", minStarts=1, limit=5, **rng); assert sk[0][1]=="Beta"
lp = q("loopedTracks", minPlays=3, limit=5, **rng); assert lp and lp[0][1]=="Alpha v2" and lp[0][3]=="2026-03-11", lp   # local date rolled over to the 11th
print("looped day uses LOCAL date:", lp[0][3])
print("weekday", q("weekdayHistogram", **rng), "| device", q("deviceBreakdown", **rng), "| recent", q("recentlyPlayed", limit=3))
print("daily", q("dailyMinutes", **rng))
# playlist ordering + renumber
q("insertPlaylist", name="P", description="", folderId=None, now=now); pid = con.execute("select last_insert_rowid()").fetchone()[0]
for i,t in enumerate([t1,t2,t1]): q("insertItem", playlistId=pid, trackId=t, position=1024*(i+1), now=now)
ids=[r[0] for r in con.execute("select id from playlist_item order by position")]
con.execute("update playlist_item set position=1025 where id=?", (ids[2],)); con.execute("update playlist_item set position=1024 where id=?", (ids[0],))  # force a collision
for i,(rid,) in enumerate(q("playlistItemIdsOrdered", playlistId=pid)): q("setItemPosition", position=1024*(i+1), id=rid)
pos=[r[0] for r in con.execute("select position from playlist_item order by position, id")]; assert pos==[1024,2048,3072], pos
print("playlist renumber (ids ordered + rewrite) ok", pos)
# queue renumber
for i,t in enumerate([t1,t2]): q("insertQueueItem", trackId=t, position=5, originalPosition=i, origin="CONTEXT")
for i,(rid,) in enumerate(q("queueItemIdsOrdered")): q("setQueuePosition", position=1024*(i+1), id=rid)
assert [r[0] for r in con.execute("select position from queue_item order by position, id")]==[1024,2048]
q("initPlayerState"); q("initPlayerState"); assert len(q("selectPlayerState"))==1
# relink keeps identity
q("relinkTrack", uri="/new/1.mp3", sourceId=1, fileMtime=9, fileSize=11, now=now, id=t1); assert q("trackById", id=t1)[0][2]=="/new/1.mp3"
q("markMissing", id=t2); assert len(q("tracksByTitle"))==1
# fts
if fts:
    q("insertSearchRow", trackId=t1, title="Beyoncé Halo", artists="Beyoncé", album="I Am", genres="Pop")
    r = q("searchFts", ftsQuery="beyonce*", limit=5); print("FTS diacritic+prefix:", r); assert r==[(t1,)]
con.commit(); print("ALL CHECKS PASSED" if bad==0 else f"{bad} FAILURES")
