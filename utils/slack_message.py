# API info: https://pypi.org/project/slackclient/
# pip install slackclient
# pip install slackclient==1.3.1
import argparse
import socket
from os import environ

from slackclient import SlackClient


def parse_args():
    parser = argparse.ArgumentParser(description='Slack Push Notificator. Enjoy!')
    parser.add_argument('--slack_token', type=str, default="SYS", help='The message to send.')
    parser.add_argument('--msg', type=str, required=True, help='The message to send.')
    parser.add_argument('--channel', type=str, default='experimentos',
                        help='The slack channel where we send the message.')
    aux = parser.parse_args()
    arguments = [aux.slack_token, aux.msg, aux.channel]
    return arguments


slack_token, msg, channel = parse_args()

if slack_token == "SYS":
    if environ.get('SLACK_TOKEN') is not None:
        slack_token = environ.get('SLACK_TOKEN')
    else:
        assert False, "Please set the environment variable SLACK_TOKEN if you want Slack notifications."


def slack_message(slack_token, message, channel):
    token = slack_token
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel,
                text=message, username='LVSC Experiments Bot',
                icon_emoji=':robot_face:')


msg = "[{}] {}".format(socket.gethostname().upper(), msg)
slack_message(slack_token, msg, channel)
