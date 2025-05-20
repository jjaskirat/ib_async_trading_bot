# Async Trading Bot with Stable Baselines3 & IB_Async 📈🤖

This project implements a trading bot that leverages **Stable Baselines3** for reinforcement learning-based trading decisions and **`ib_async`** for asynchronous communication with the Interactive Brokers API.
The bot is designed to download market data every 5 seconds and trigger trade actions immediately upon receiving new data.

---

## 🚀 Features

* **Reinforcement Learning Driven:** Utilizes a pre-trained model from Stable Baselines3 to make trading decisions.
* **Asynchronous Operations:** Employs `ib_async` for non-blocking communication with Interactive Brokers, allowing for efficient data handling and order placement.
* **Real-time Data Feed:** Downloads new market data at a 5-second interval.
* **Rapid Trade Execution:** Triggers trade actions (buy/sell/hold) as soon as new data is processed by the model.
* **Interactive Brokers Integration:** Connects directly to the IB Trader Workstation (TWS) or IB Gateway.

---

## 🛠️ Technologies Used

* **Python 3.x**
* **Stable Baselines3:** For loading and using reinforcement learning models.
* **`ib_async`:** For interacting with the Interactive Brokers API.
* **Pandas:** For data manipulation (likely for handling market data).
* **NumPy:** For numerical operations.

---
## Video Demo
https://github.com/user-attachments/assets/1098aab4-d620-49ad-879e-53da171c40bc

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jjaskirat/ib_async_trading_bot.git](https://github.com/jjaskirat/ib_async_trading_bot.git)
    cd ib_async_trading_bot
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -e .
    ```

3.  **Interactive Brokers Setup:**
    * Ensure you have **Trader Workstation (TWS)** or **IB Gateway** installed and running.
    * [https://www.interactivebrokers.com/en/trading/ibgateway-stable.php](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
    * Configure TWS/Gateway to allow API connections:
        * Go to `File > Global Configuration > API > Settings`.
        * Enable "Enable ActiveX and Socket Clients".
        * Note the **Socket port** (default is 7497 for TWS, 4002 for Gateway).
        * Add `127.0.0.1` to the "Trusted IP Addresses" if running the bot on the same machine.
     
4.  **IBC Setup**
   * Follow the Instructions [https://github.com/IbcAlpha/IBC/blob/master/userguide.md](https://github.com/IbcAlpha/IBC/blob/master/userguide.md)
   * You need to run the files `/opt/ibc/twsstart.sh` and `/opt/ibc/stop.sh` without sudo
       * Open a Terminal window and type: `sudo vim /etc/sudoers`
       * In the bottom of the file, add the following lines:
         ```bash
         $USER ALL=(ALL) NOPASSWD: /opt/ibc/twsstart.sh
         $USER ALL=(ALL) NOPASSWD: /opt/ibc/stop.sh
         ```
