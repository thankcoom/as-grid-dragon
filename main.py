# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
AS 刷怪籠 - 網格交易系統 MAX 版本
================================
程式入口點
"""

from ui.menu import MainMenu


def main():
    """主程式入口"""
    menu = MainMenu()
    menu.main_menu()


if __name__ == "__main__":
    main()
