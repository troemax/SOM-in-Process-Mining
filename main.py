if __name__ == "__main__":
    import panel as pn
    import real_main
    pn.serve(real_main.template, port=5006)