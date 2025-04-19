from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/")


mySoccerNetDownloader.password = "s0cc3rn3t"

mySoccerNetDownloader.downloadGames(files=["Labels-v3.json", ], split=["train","valid","test"])
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])
