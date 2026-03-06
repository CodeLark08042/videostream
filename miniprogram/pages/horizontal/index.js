const app = getApp()

Page({
  data: {
    videoFeedUrl: "",
    videoRawUrl: "",
    avgSpeed: 0,
    density: 0,
    vehicleCount: 0,
    trafficStatus: "检测中",
    videoFps: 0,
    inferenceFps: 0,
    vehicleList: []
  },

  timer: null,

  onLoad() {
    const baseUrl = app.globalData.serverUrl;
    this.setData({
      videoFeedUrl: `${baseUrl}/video_feed`,
      videoRawUrl: `${baseUrl}/video_raw`
    });
    
    // 开始轮询数据
    this.startPolling();
  },

  onUnload() {
    this.stopPolling();
  },

  onHide() {
    this.stopPolling();
  },

  onShow() {
    if (!this.timer) {
      this.startPolling();
    }
  },

  startPolling() {
    this.fetchData();
    this.timer = setInterval(() => {
      this.fetchData();
    }, 1000);
  },

  stopPolling() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  },

  fetchData() {
    const baseUrl = app.globalData.serverUrl;
    wx.request({
      url: `${baseUrl}/api/horizontal`,
      method: 'GET',
      success: (res) => {
        if (res.statusCode === 200) {
          const { data, fps } = res.data;
          this.setData({
            avgSpeed: data.avg_speed,
            density: data.density,
            vehicleCount: data.vehicle_count,
            trafficStatus: data.status,
            vehicleList: data.vehicle_list || [],
            videoFps: fps.video_fps,
            inferenceFps: fps.inference_fps
          });
        }
      },
      fail: (err) => {
        console.error("Fetch data failed:", err);
      }
    });
  }
})
