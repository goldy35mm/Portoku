import os
from datetime import date, timedelta
import requests, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Leo Portfolio Lab",page_icon="📈",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp {background: radial-gradient(ellipse at 15% 0%, rgba(19,91,112,.22), transparent 38%), #08111e;}
[data-testid="stHeader"] {background:rgba(8,17,30,.78);}
.block-container {max-width:1600px;padding-top:1.25rem;padding-bottom:2.5rem;}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em;}
h1 {font-size:2.25rem!important;font-weight:700!important;}
[data-testid="stMetric"] {background:linear-gradient(145deg,rgba(20,37,57,.96),rgba(13,26,42,.96));border:1px solid rgba(117,170,194,.19);padding:17px 18px;border-radius:15px;box-shadow:0 8px 24px rgba(0,0,0,.15);min-height:112px;}
[data-testid="stMetricLabel"] {color:#9bb0c6;font-size:.83rem;font-weight:600;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;color:#eff7ff;}
[data-testid="stMetricDelta"] {font-size:.82rem;}
[data-testid="stTabs"] button {font-weight:650;border-radius:10px 10px 0 0;padding:12px 16px;}
[data-testid="stDataFrame"],[data-testid="stTable"] {border:1px solid rgba(117,170,194,.17);border-radius:12px;overflow:hidden;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-color:rgba(117,170,194,.2)!important;border-radius:15px!important;}
.stButton>button,.stDownloadButton>button {border-radius:10px;font-weight:650;border:1px solid rgba(92,190,205,.35);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0b1727,#0a1422);}
[data-testid="stCaptionContainer"] {color:#8ea4ba;}
hr {border-color:rgba(117,170,194,.18);}
.hero {padding:24px 26px;border-radius:18px;border:1px solid rgba(82,190,205,.23);background:linear-gradient(115deg,rgba(12,44,61,.98),rgba(17,30,49,.95) 60%,rgba(23,48,65,.92));margin:4px 0 20px;}
.hero-kicker {color:#61d6d0;font-size:.76rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;}
.hero-title {font-family:'Space Grotesk',sans-serif;font-size:1.75rem;font-weight:700;color:#f0f8ff;margin-top:5px;}
.hero-sub {color:#a6bed0;font-size:.92rem;margin-top:4px;}
.section-kicker {font-size:.72rem;font-weight:700;letter-spacing:.12em;color:#65c8ce;text-transform:uppercase;margin-bottom:4px;}
</style>
""",unsafe_allow_html=True)
st.markdown("""
<div class="hero">
 <div class="hero-kicker">IDX PORTFOLIO INTELLIGENCE</div>
 <div class="hero-title">Leo Portfolio Lab</div>
 <div class="hero-sub">Market overview · Technical signals · LSTM scenarios · Portfolio risk</div>
</div>
""",unsafe_allow_html=True)

DEFAULT=[{"Kode":"ADRO","Lot":261,"Avg Beli":2066.36},{"Kode":"ADMR","Lot":355,"Avg Beli":1589.32},{"Kode":"ESSA","Lot":1000,"Avg Beli":612.68},{"Kode":"MEDC","Lot":82,"Avg Beli":1471.10}]
if "holdings" not in st.session_state: st.session_state.holdings=pd.DataFrame(DEFAULT)
if "cash" not in st.session_state: st.session_state.cash=20027683.0
if "models" not in st.session_state: st.session_state.models={}
try: KEY=st.secrets.get("ZAPI_API_KEY",os.getenv("ZAPI_API_KEY",""))
except Exception: KEY=os.getenv("ZAPI_API_KEY","")
URL="https://api.zapi.ink/v1/finance:idx/stock-history"

def rows_from(x):
    if isinstance(x,list):
        if x and all(isinstance(v,dict) for v in x): return x
        for v in x:
            r=rows_from(v)
            if r:return r
    if isinstance(x,dict):
        for k in ("data","results","result","items","rows","stock_history"):
            if k in x:
                r=rows_from(x[k])
                if r:return r
        for v in x.values():
            if isinstance(v,(list,dict)):
                r=rows_from(v)
                if r:return r
    return []

@st.cache_data(ttl=300,show_spinner=False)
def get_history(code,start,end,key):
    res=requests.get(URL,headers={"x-api-key":key,"accept":"application/json"},
                     params={"code":code,"from":start,"to":end,"length":2000},timeout=(20,60))
    res.raise_for_status(); rows=rows_from(res.json())
    if not rows: raise ValueError("Respons API kosong/tidak dikenali")
    d=pd.DataFrame(rows); d.columns=[str(c).lower().strip() for c in d.columns]
    aliases={"date":["date","datetime","timestamp","trading_date","trade_date"],
             "open":["open","o"],"high":["high","h"],"low":["low","l"],
             "close":["close","c","price","last"],"volume":["volume","vol","v"]}
    ren={}
    for target,names in aliases.items():
        for c in d.columns:
            if c in names: ren[c]=target; break
    d=d.rename(columns=ren)
    if "close" not in d or "date" not in d: raise ValueError(f"Kolom tanggal/close tidak ada: {list(d.columns)}")
    raw=d.date
    if pd.api.types.is_numeric_dtype(raw):
        med=pd.to_numeric(raw,errors="coerce").dropna().median()
        d.date=pd.to_datetime(raw,unit="ms" if med>1e11 else "s",errors="coerce",utc=True).dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    else: d.date=pd.to_datetime(raw,errors="coerce",utc=True).dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    for c in ("open","high","low","close","volume"):
        if c in d: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=["date","close"]).sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)

def enrich(d):
    d=d.copy(); c=d.close; delta=c.diff()
    for n in (5,10,20,50,100,200): d[f"MA{n}"]=c.rolling(n).mean()
    d["EMA9"]=c.ewm(span=9,adjust=False).mean(); d["EMA21"]=c.ewm(span=21,adjust=False).mean()
    up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    dn=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d["RSI14"]=100-100/(1+up/dn.replace(0,np.nan))
    d["Return %"]=c.pct_change()*100; d["Volatility20"]=d["Return %"].rolling(20).std()
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    d["MACD"]=e12-e26; d["MACD Signal"]=d.MACD.ewm(span=9,adjust=False).mean()
    if {"high","low"}.issubset(d.columns):
        prev=c.shift()
        tr=pd.concat([(d.high-d.low).abs(),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
        d["ATR14"]=tr.rolling(14).mean()
    else: d["ATR14"]=np.nan
    return d

def rp(x): return f"Rp {x:,.0f}" if pd.notna(x) and np.isfinite(x) else "—"

with st.sidebar:
    st.header("Pengaturan")
    start=st.date_input("Mulai data",date.today()-timedelta(days=365*5))
    end=st.date_input("Akhir data",date.today())
    st.session_state.cash=st.number_input("Cash (Rp)",min_value=0.0,value=float(st.session_state.cash),step=1000000.0)
    if st.button("🔄 Refresh ZAPI",use_container_width=True): get_history.clear(); st.session_state.models={}; st.rerun()
if not KEY: st.error("Isi ZAPI_API_KEY pada Streamlit Cloud → Settings → Secrets."); st.stop()
if start>=end: st.error("Tanggal akhir harus setelah tanggal mulai."); st.stop()

st.subheader("🧾 Kelola saham")
edited=st.data_editor(st.session_state.holdings,num_rows="dynamic",use_container_width=True,hide_index=True,
    column_config={"Kode":st.column_config.TextColumn("Kode BEI",required=True),
    "Lot":st.column_config.NumberColumn("Lot",min_value=0,step=1,required=True),
    "Avg Beli":st.column_config.NumberColumn("Average beli (Rp)",min_value=0,step=1,required=True)},key="editor")
if st.button("💾 Simpan perubahan"):
    edited["Kode"]=edited.Kode.astype(str).str.upper().str.strip()
    edited=edited[(edited.Kode!="")&(pd.to_numeric(edited.Lot,errors="coerce").fillna(0)>0)&(pd.to_numeric(edited["Avg Beli"],errors="coerce").fillna(0)>0)]
    if edited.Kode.duplicated().any(): st.error("Kode tidak boleh duplikat.")
    else: st.session_state.holdings=edited.reset_index(drop=True); st.session_state.models={}; st.rerun()
with st.form("new_stock"):
    x,y,z=st.columns([2,1,2]); code=x.text_input("Tambah kode BEI",placeholder="BBCA").upper().strip()
    lot=y.number_input("Lot",min_value=1,value=1); avg=z.number_input("Average beli",min_value=1.0,value=1000.0)
    add=st.form_submit_button("＋ Tambah saham")
if add:
    if not code: st.warning("Masukkan kode saham.")
    elif code in st.session_state.holdings.Kode.astype(str).str.upper().tolist(): st.warning("Kode sudah ada.")
    else: st.session_state.holdings=pd.concat([st.session_state.holdings,pd.DataFrame([{"Kode":code,"Lot":lot,"Avg Beli":avg}])],ignore_index=True); st.session_state.models={}; st.rerun()

h=st.session_state.holdings.copy(); h["Kode"]=h.Kode.astype(str).str.upper().str.strip()
h=h[(h.Kode!="")&(pd.to_numeric(h.Lot,errors="coerce").fillna(0)>0)]
if h.empty: st.info("Tambahkan saham untuk memulai."); st.stop()
hist={}; errors={}
with st.spinner("Mengambil data historis dari ZAPI..."):
    for ticker in h.Kode:
        try: hist[ticker]=enrich(get_history(ticker,start.isoformat(),end.isoformat(),KEY))
        except Exception as e: errors[ticker]=str(e)
for ticker,e in errors.items(): st.warning(f"{ticker}: {e}")

data=[]
for _,p in h.iterrows():
    d=hist.get(p.Kode); price=float(d.close.iloc[-1]) if d is not None and len(d) else np.nan
    qty=float(p.Lot)*100; cost=qty*float(p["Avg Beli"]); value=qty*price if np.isfinite(price) else np.nan
    data.append({"Kode":p.Kode,"Lot":p.Lot,"Saham":qty,"Avg Beli":p["Avg Beli"],"Harga":price,
                 "Modal":cost,"Nilai Pasar":value,"P&L":value-cost if np.isfinite(value) else np.nan,
                 "P&L %":(value-cost)/cost*100 if cost and np.isfinite(value) else np.nan,
                 "Tanggal Data":d.date.iloc[-1].strftime("%d-%m-%Y") if d is not None and len(d) else "N/A"})
port=pd.DataFrame(data); good=port.dropna(subset=["Nilai Pasar"])
mv=good["Nilai Pasar"].sum(); pnl=good.P&L.sum(); equity=st.session_state.cash+mv; cost=good.Modal.sum()
st.divider(); st.subheader("💼 Portfolio overview")
a,b,c,d=st.columns(4); a.metric("Total ekuitas",rp(equity)); b.metric("Nilai pasar",rp(mv))
c.metric("Cash",rp(st.session_state.cash)); d.metric("Floating P&L",rp(pnl),f"{pnl/cost:+.2%}" if cost else None)
if len(good)<len(port): st.info("Ekuitas hanya menghitung saham dengan data harga yang berhasil diambil.")
v=port.copy()
for col in ("Avg Beli","Harga","Modal","Nilai Pasar","P&L"): v[col]=v[col].map(rp)
v["P&L %"]=v["P&L %"].map(lambda q:f"{q:+.2f}%" if pd.notna(q) else "—")
st.dataframe(v,use_container_width=True,hide_index=True)
if len(good):
    pie=go.Figure(go.Pie(labels=good.Kode,values=good["Nilai Pasar"],hole=.58))
    pie.update_layout(title="Komposisi nilai pasar",height=330,margin=dict(t=50,b=10))
    st.plotly_chart(pie,use_container_width=True)

t1,t2,t3=st.tabs(["📊 Teknikal & historis","🧠 LSTM","🎯 Simulasi & export"])
available=[x for x in h.Kode if x in hist]
with t1:
    if available:
        ticker=st.selectbox("Pilih saham",available,key="chart_ticker"); d=hist[ticker]; last=d.iloc[-1]
        q1,q2,q3,q4=st.columns(4); q1.metric("Harga terakhir",rp(last.close))
        q2.metric("RSI 14",f"{last.RSI14:.1f}" if pd.notna(last.RSI14) else "—")
        q3.metric("ATR 14",rp(last.ATR14)); q4.metric("Volatilitas 20",f"{last.Volatility20:.2f}%" if pd.notna(last.Volatility20) else "—")
        mode=st.radio("Grafik",["Candlestick + MA","Close + EMA","Return harian"],horizontal=True)
        if mode=="Candlestick + MA" and {"open","high","low"}.issubset(d.columns):
            f=go.Figure(go.Candlestick(x=d.date,open=d.open,high=d.high,low=d.low,close=d.close,name=ticker))
            for col in ("MA20","MA50","MA200"): f.add_trace(go.Scatter(x=d.date,y=d[col],name=col))
        elif mode=="Close + EMA":
            f=go.Figure(go.Scatter(x=d.date,y=d.close,name="Close"))
            for col in ("EMA9","EMA21"): f.add_trace(go.Scatter(x=d.date,y=d[col],name=col))
        else: f=go.Figure(go.Bar(x=d.date,y=d["Return %"],name="Return %"))
        f.update_layout(height=470,xaxis_rangeslider_visible=False,margin=dict(t=25,b=10),
                        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Sans",color="#d8e7f5"),legend=dict(orientation="h",y=1.08))
        st.plotly_chart(f,use_container_width=True)
        u,w=st.columns(2)
        with u:
            f=go.Figure(go.Scatter(x=d.date,y=d.RSI14,name="RSI")); f.add_hline(y=70,line_dash="dash"); f.add_hline(y=30,line_dash="dash")
            f.update_layout(title="RSI 14",height=280,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(f,use_container_width=True)
        with w:
            f=go.Figure(); f.add_trace(go.Scatter(x=d.date,y=d.MACD,name="MACD")); f.add_trace(go.Scatter(x=d.date,y=d["MACD Signal"],name="Signal"))
            f.update_layout(title="MACD",height=280,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(f,use_container_width=True)
        st.caption(f"{ticker}: {len(d)} baris, {d.date.iloc[0]:%d-%m-%Y} s.d. {d.date.iloc[-1]:%d-%m-%Y}. Tanggal mengikuti data terakhir API.")
        with st.expander("Tabel historis + indikator"): st.dataframe(d.sort_values("date",ascending=False),use_container_width=True,hide_index=True)
with t2:
    if available:
        ticker=st.selectbox("Saham untuk LSTM",available,key="lstm_ticker"); d=hist[ticker]
        st.write(f"Data training: {len(d)} sesi, {d.date.iloc[0]:%d-%m-%Y} sampai {d.date.iloc[-1]:%d-%m-%Y}")
        st.warning("Skenario statistik, bukan harga pasti atau rekomendasi transaksi. LSTM mempelajari pola historis yang dapat berubah ketika kondisi pasar berganti.")
        epochs=st.slider("Epoch maksimum",10,80,35,5)
        if st.button("🧠 Latih / perbarui LSTM",type="primary"):
            try:
                from sklearn.preprocessing import MinMaxScaler
                from sklearn.metrics import mean_absolute_error,mean_squared_error
                from tensorflow.keras import Sequential
                from tensorflow.keras.layers import LSTM,Dense,Dropout
                from tensorflow.keras.callbacks import EarlyStopping
                raw=d[["close"]].copy()
                for n in (5,20,60): raw[f"y{n}"]=raw.close.shift(-n)/raw.close-1
                raw=raw.dropna().reset_index(drop=True)
                if len(raw)<300: raise ValueError(f"Perlu minimal 300 baris berlabel, tersedia {len(raw)}")
                look=40; split=int(len(raw)*.8); scaler=MinMaxScaler().fit(raw[["close"]].iloc[:split])
                scaled=scaler.transform(raw[["close"]]); X=[]; Y=[]
                target=raw[["y5","y20","y60"]].values
                for i in range(look,len(raw)): X.append(scaled[i-look:i,0]); Y.append(target[i])
                X=np.array(X,dtype="float32").reshape(-1,look,1); Y=np.array(Y,dtype="float32")
                cut=max(1,split-look); Xtr,Ytr=X[:cut],Y[:cut]; Xte,Yte=X[cut:],Y[cut:]
                if len(Xte)<20: raise ValueError("Sampel uji terlalu sedikit.")
                model=Sequential([LSTM(48,input_shape=(look,1)),Dropout(.2),Dense(24,activation="relu"),Dense(3)])
                model.compile(optimizer="adam",loss="huber")
                model.fit(Xtr,Ytr,validation_split=.15,epochs=epochs,batch_size=32,verbose=0,
                          callbacks=[EarlyStopping(patience=6,restore_best_weights=True)])
                p=model.predict(Xte,verbose=0); residual=Yte-p
                pred=model.predict(scaler.transform(d[["close"]].tail(look)).reshape(1,look,1),verbose=0)[0]
                current=float(d.close.iloc[-1]); fr=[]; met=[]
                for j,n in enumerate((5,20,60)):
                    lo,hi=np.quantile(residual[:,j],[.1,.9])
                    fr.append({"Horizon":f"{n} sesi","Estimasi return %":pred[j]*100,
                               "Skenario bawah":current*(1+pred[j]+lo),"Estimasi tengah":current*(1+pred[j]),
                               "Skenario atas":current*(1+pred[j]+hi)})
                    met.append({"Horizon":f"{n} sesi","MAE return (pp)":mean_absolute_error(Yte[:,j],p[:,j])*100,
                                "RMSE return (pp)":np.sqrt(mean_squared_error(Yte[:,j],p[:,j]))*100,
                                "Akurasi arah %":np.mean(np.sign(Yte[:,j])==np.sign(p[:,j]))*100})
                st.session_state.models[ticker]={"forecast":pd.DataFrame(fr),"metrics":pd.DataFrame(met),
                                                 "asof":d.date.iloc[-1].strftime("%d-%m-%Y")}
                st.success("Model selesai dilatih.")
            except Exception as e: st.error(f"LSTM gagal: {e}. Periksa TensorFlow dan kecukupan data.")
        result=st.session_state.models.get(ticker)
        if result:
            st.caption(f"Model dilatih berdasarkan data sampai {result['asof']}.")
            f=result["forecast"].copy()
            f["Estimasi return %"]=f["Estimasi return %"].map(lambda x:f"{x:+.2f}%")
            for col in ("Skenario bawah","Estimasi tengah","Skenario atas"): f[col]=f[col].map(rp)
            st.subheader("Skenario harga"); st.dataframe(f,use_container_width=True,hide_index=True)
            st.caption("Rentang menggunakan kuantil residual 10–90% dari holdout waktu; belum merupakan interval prediksi terkalibrasi.")
            st.subheader("Evaluasi holdout kronologis")
            st.dataframe(result["metrics"].style.format("{:.2f}",subset=["MAE return (pp)","RMSE return (pp)","Akurasi arah %"]),use_container_width=True,hide_index=True)
            st.caption("MAE/RMSE dalam poin persentase return. Evaluasi holdout historis bukan jaminan hasil mendatang.")
        else: st.info("Klik Latih / perbarui LSTM untuk menghitung skenario.")
with t3:
    st.caption("Ubah harga secara hipotetis untuk melihat sensitivitas nilai portofolio.")
    sim=[]
    for _,r in port.iterrows():
        ch=st.slider(f"{r.Kode} (%)",-50,100,0,1,key=f"shock_{r.Kode}")
        if pd.notna(r.Harga): sim.append({"Kode":r.Kode,"Perubahan %":ch,"Harga simulasi":r.Harga*(1+ch/100),"Nilai simulasi":r.Saham*r.Harga*(1+ch/100)})
    if sim:
        sdf=pd.DataFrame(sim); eq=st.session_state.cash+sdf["Nilai simulasi"].sum()
        st.metric("Ekuitas simulasi",rp(eq),f"{eq-equity:+,.0f} Rp")
        st.dataframe(sdf,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Unduh ringkasan CSV",port.to_csv(index=False).encode("utf-8-sig"),"portfolio.csv","text/csv")
    if hist:
        allhist=pd.concat([v.assign(Kode=k) for k,v in hist.items()],ignore_index=True)
        st.download_button("⬇️ Unduh historis + indikator CSV",allhist.to_csv(index=False).encode("utf-8-sig"),"idx_history.csv","text/csv")
st.divider()
st.caption("ZAPI IDX • Lot diasumsikan 100 saham. Belum termasuk biaya, pajak, dividen, corporate action, dan rights issue. Verifikasi harga terakhir serta input portofolio terhadap aplikasi broker.")
