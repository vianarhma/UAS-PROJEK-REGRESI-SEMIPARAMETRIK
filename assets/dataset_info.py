"""
Dataset Information Module
Berisi informasi lengkap tentang dataset yang digunakan
"""

DATASET_INFO = {
    "name": "Cryptocurrency Price Data (KuCoin)",
    "source": "KuCoin Exchange API",
    "interval": "1 Hour (1H)",
    "description": """
    Dataset ini berisi data historis harga cryptocurrency dengan interval 1 jam.
    Data mencakup harga pembukaan (open), tertinggi (high), terendah (low), 
    penutupan (close), dan volume transaksi untuk setiap periode 1 jam.
    Data diperoleh dari platform KuCoin sebagai salah satu exchange terkemuka dunia.
    """,
    "use_case": "Prediksi harga cryptocurrency 1 jam ke depan menggunakan metode Nadaraya-Watson Kernel dan LightGBM"
}

COLUMN_DESCRIPTIONS = {
    "datetime": {
        "name": "Datetime",
        "type": "DateTime",
        "description": "Timestamp untuk setiap periode 1 jam",
        "example": "2024-01-01 00:00:00",
        "role": "Index temporal"
    },
    "open": {
        "name": "Open Price",
        "type": "Float",
        "description": "Harga pembukaan pada awal periode 1 jam",
        "unit": "USD",
        "role": "Feature (Variabel Eksogen)"
    },
    "high": {
        "name": "High Price",
        "type": "Float",
        "description": "Harga tertinggi yang dicapai selama periode 1 jam",
        "unit": "USD",
        "role": "Feature (Variabel Eksogen)"
    },
    "low": {
        "name": "Low Price",
        "type": "Float",
        "description": "Harga terendah yang tercatat selama periode 1 jam",
        "unit": "USD",
        "role": "Feature (Variabel Eksogen)"
    },
    "close": {
        "name": "Close Price",
        "type": "Float",
        "description": "Harga penutupan pada akhir periode 1 jam (TARGET PREDIKSI)",
        "unit": "USD",
        "role": "Target Variable (Y)"
    },
    "volume": {
        "name": "Trading Volume",
        "type": "Float",
        "description": "Total volume transaksi SOL yang diperdagangkan dalam periode 1 jam",
        "unit": "SOL",
        "role": "Feature (Variabel Eksogen)"
    }
}

SOLANA_INFO = {
    "name": "KuCoin",
    "symbol": "KCS",
    "launch_year": 2017,
    "founder": "KuCoin Team",
    "blockchain_type": "Cryptocurrency Exchange Platform",
    "consensus": "Centralized Exchange",
    "description": """
    KuCoin adalah platform mata uang kripto terkemuka dunia, yang dipercaya oleh lebih dari 40 juta pengguna di lebih dari 200 negara dan wilayah. KuCoin menawarkan beragam pilihan perdagangan, termasuk perdagangan spot, leverage, opsi, dan derivatif, serta layanan seperti DeFi, peminjaman, dan penambangan. KuCoin menawarkan pengalaman perdagangan kripto yang sepenuhnya inklusif, memberikan akses ke lebih dari 1.000 koin, konversi token bebas biaya, leverage hingga 10x, dan lebih dari 450+ derivatif kripto. Dengan fitur seperti copy trading dan staking fleksibel, platform kami memberdayakan Anda untuk menghasilkan profit, melakukan hedging, dan bertumbuh secara efisien.
    """,
    "key_features": [
        "🌍 Global Reach: 40+ juta pengguna di 200+ negara",
        "💰 Beragam Pilihan: Spot, leverage, opsi, derivatif",
        "🔄 DeFi Services: Peminjaman, penambangan, staking",
        "📈 Leverage Trading: Hingga 10x leverage",
        "🪙 1000+ Koin: Akses ke ribuan cryptocurrency",
        "💱 Konversi Bebas Biaya: Token conversion tanpa fee",
        "📊 450+ Derivatif: Beragam instrumen derivatif",
        "👥 Copy Trading: Ikuti trader sukses",
        "🔄 Staking Fleksibel: Berbagai opsi staking"
    ],
    "use_cases": [
        "💱 Spot Trading: Beli/jual cryptocurrency langsung",
        "📈 Leverage Trading: Trading dengan margin",
        "🔮 Derivatives: Opsi dan futures trading",
        "💰 DeFi Services: Peminjaman dan staking",
        "⛏️ Mining: Cloud mining services",
        "📊 Copy Trading: Ikuti strategi trader pro",
        "🔄 Token Conversion: Konversi antar token",
        "💳 Payment Solutions: Integrasi pembayaran crypto"
    ],
    "advantages": [
        "Platform terpercaya dengan 40+ juta pengguna",
        "Beragam pilihan perdagangan dan layanan",
        "Akses ke 1000+ cryptocurrency",
        "Fitur inovatif seperti copy trading",
        "Biaya rendah dan konversi bebas fee",
        "Leverage hingga 10x untuk profit maksimal",
        "Staking fleksibel untuk passive income",
        "Dukungan multi-bahasa dan 24/7"
    ],
    "technology": {
        "Trading_Engine": """
        KuCoin menggunakan trading engine canggih yang mendukung high-frequency trading
        dengan latency rendah dan eksekusi order yang cepat. Platform ini mampu
        menangani jutaan transaksi per hari dengan keamanan tinggi.
        """,
        "Security": """
        KuCoin mengimplementasikan multi-layer security system termasuk cold wallet storage,
        two-factor authentication, dan insurance fund untuk melindungi aset pengguna.
        Platform ini telah lulus audit keamanan dari firma terkemuka.
        """
    }
}

RESEARCH_OBJECTIVES = {
    "main_objective": """
    Membandingkan performa dua metode prediksi time series (Nadaraya-Watson Kernel 
    dan LightGBM) dalam memprediksi harga cryptocurrency 1 jam ke depan.
    """,
    "specific_objectives": [
        "Mengimplementasikan metode regresi nonparametrik Nadaraya-Watson dengan kernel Gaussian",
        "Mengimplementasikan metode machine learning LightGBM untuk time series forecasting",
        "Melakukan feature engineering dengan lag features untuk prediksi 1 jam ke depan",
        "Mengevaluasi performa kedua model menggunakan metrik MSE, RMSE, MAPE, dan R²",
        "Membandingkan kelebihan dan kekurangan masing-masing metode",
        "Memberikan rekomendasi metode terbaik untuk prediksi harga cryptocurrency"
    ],
    "research_questions": [
        "Metode mana yang lebih akurat dalam memprediksi harga cryptocurrency 1 jam ke depan?",
        "Bagaimana performa regresi semiparametrik dibandingkan dengan machine learning?",
        "Variabel mana yang paling berpengaruh terhadap prediksi harga?",
        "Apakah metode nonparametrik dapat menangkap volatilitas cryptocurrency?"
    ],
    "benefits": [
        "📊 Memberikan insight untuk trading decision",
        "🔬 Kontribusi akademik dalam bidang regresi semiparametrik",
        "💡 Pemahaman tentang karakteristik time series cryptocurrency",
        "🎯 Evaluasi metode prediksi untuk asset volatil"
    ],
    "hypothesis": """
    Hipotesis penelitian ini adalah bahwa LightGBM akan memberikan akurasi 
    prediksi yang lebih tinggi dibandingkan Nadaraya-Watson Kernel karena 
    kemampuannya dalam menangkap non-linear patterns dan interaksi antar variabel.
    Namun, Nadaraya-Watson memiliki keunggulan dalam fleksibilitas dan tidak 
    memerlukan asumsi distribusi data.
    """
}

DATA_SPLIT_INFO = {
    "train_percentage": 80,
    "test_percentage": 20,
    "method": {
        "nadaraya_watson": "Random sampling (shuffle=True) untuk menghindari bias temporal",
        "lightgbm": "Temporal split (shuffle=False) untuk menjaga urutan waktu"
    },
    "rationale": """
    - Nadaraya-Watson: Menggunakan random split karena metode ini tidak bergantung 
      pada urutan temporal dan lebih fokus pada jarak antar observasi
    - LightGBM: Menggunakan temporal split untuk mensimulasikan prediksi real-time 
      dimana model hanya mengetahui data masa lalu
    """
}