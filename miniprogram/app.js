// app.js
App({
  onLaunch() {
    console.log('App Launch');
  },
  globalData: {
    // 自动判断环境：
    // 1. 开发者工具 (devtools) -> 使用 http://127.0.0.1:5000 (避免 localhost 解析问题)
    // 2. 真机 (Android/iOS) -> 使用 Cloudflare 隧道地址 (请每次运行后更新下方的链接)
    serverUrl: "https://u900342-a9b7-cbc99456.westb.seetacloud.com:8443"
    //serverUrl: wx.getSystemInfoSync().platform === 'devtools' 
     // ? "http://127.0.0.1:5000" 
      //: "https://u900342-a9b7-cbc99456.westb.seetacloud.com:8443" 
  }
})
