const app = getApp()

Page({
  data: {
    videoUrl: "",
    mainStatus: "--",
    mainStatusClass: "normal",
    mergeGap: "--",
    riskLevel: "--",
    advice: "分析中...",
    riskClass: "low", // low, medium, high
    canMerge: false,
    showVoiceAlert: false,
    showVibration: false
  },

  timer: null,
  lastAdvice: "",

  onLoad() {
    this.setData({
      videoUrl: `${app.globalData.serverUrl}/video_feed`
    });
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
    this.fetchData(); // 立即执行一次
    this.timer = setInterval(() => {
      this.fetchData();
    }, 1000); // 每秒轮询
  },

  stopPolling() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  },

  fetchData() {
    wx.request({
      url: `${app.globalData.serverUrl}/api/vertical`,
      method: 'GET',
      success: (res) => {
        if (res.statusCode === 200) {
          const { data } = res.data; // API 返回的数据结构可能需要调整

          let riskClass = "low";
          if (data.risk_level === "中") riskClass = "medium";
          if (data.risk_level === "高") riskClass = "high";

          let mainStatusClass = "normal";
          if (data.main_road_status === "缓行") mainStatusClass = "slow";
          if (data.main_road_status === "拥堵") mainStatusClass = "congested";

          this.setData({
            mainStatus: data.main_road_status,
            mainStatusClass: mainStatusClass,
            mergeGap: data.merge_gap.toFixed(1),
            riskLevel: data.risk_level,
            advice: data.advice,
            riskClass: riskClass,
            canMerge: data.can_merge
          });

          // 触发多模态反馈
          if (data.advice !== this.lastAdvice) {
            this.triggerAlerts(data.advice, riskClass);
            this.lastAdvice = data.advice;
          }
        }
      },
      fail: (err) => {
        console.error("Vertical data fetch failed:", err);
      }
    });
  },

  triggerAlerts(advice, riskClass) {
    // 模拟语音播报
    if (advice && advice !== '分析中...') {
      this.setData({ showVoiceAlert: true });
      setTimeout(() => this.setData({ showVoiceAlert: false }), 2000);
    }

    // 模拟震动反馈
    if (riskClass === 'high') {
      wx.vibrateLong();
      this.setData({ showVibration: true });
      setTimeout(() => this.setData({ showVibration: false }), 1500);
    } else if (riskClass === 'medium') {
      wx.vibrateShort();
    }
  }
});
