// ==UserScript==
// @name         Instagram Followers
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://www.instagram.com/*
// @icon         https://www.google.com/s2/favicons?domain=tampermonkey.net
// @require      https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js
// @require      https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js
// @require      https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.0/FileSaver.min.js
// @require      https://gist.github.com/raw/2625891/waitForKeyElements.js
// @resource     jqUI_CSS https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/themes/base/jquery-ui.min.css
// @grant        GM_getResourceURL
// @grant        GM_getResourceText
// @grant        GM_addStyle
// ==/UserScript==

const styleSheet = `
#menu {
    width: 120px;
    background-color: white !important;
    position: absolute;
    top: 74px;
    right: 0;
}
.ui-dialog, .ui-menu {
    z-index: 9999 !important;
}
`;

const dialog = $(`
<div class="crawler-dialog" title="Loading data">
    <p>
        <div class="crawler-progressbar">
            <div class="crawler-progressbar-label">Loading...</div>
        </div>
    </p>
</div>
`);

const menu = $(`
<ul id="menu">
  <li class="menu-followers-btn"><div class="ui-state-disabled">Followers</div></li>
</ul>
`);

const progressBar = dialog.find('.crawler-progressbar');
const progressBarLabel = progressBar.find('.crawler-progressbar-label');

const maxFollowersPerAccount = 100000;
const followersPerPage = 2500;

let stopped = false;
let authorId = "";

let results = [];
let idMap = {};
let followers = []
let pageInfo;
let media;

function addStyles() {
    console.log('adding additional styles');
    GM_addStyle(styleSheet);
    var jqUI_CssSrc = GM_getResourceText('jqUI_CSS');
    GM_addStyle(jqUI_CssSrc);
}

function addMenu() {
    $('body').append(menu);
    menu.menu();
    menu.menu({
        select: onMenuSelected
    });
}

function addProgressDialog() {
    $('body').append(dialog);
    dialog.append(progressBar);
    progressBar.progressbar({
        value: false,
        change: () => {
            progressBarLabel.text(progressBar.progressbar('value') + ' / ' + progressBar.progressbar('option', 'max'));
        }
    });
    dialog.dialog({
        autoOpen: false,
        buttons: [{
            class: 'crawler-download-btn',
            text: 'Download',
            disabled: true,
            click: onDownloadClicked
        }, {
            class: 'crawler-stop-btn',
            text: 'Stop',
            click: onStopClicked
        }]
    });
}

function init() {
    addStyles();
    addMenu();
    addProgressDialog();
    if ($('meta[property="og:type"]').prop('content') === 'profile') {
        menu.find('.menu-followers-btn > div').removeClass('ui-state-disabled');
    }
}

function onMenuSelected(event, ui) {
    console.log('Menu selected!');
    console.log(event);
    console.log(ui);
    if (ui.item.hasClass('menu-followers-btn')) {
        onFollowerButtonClicked();
    }
}

function onStopClicked() {
    stopped = true;
}

function onDownloadClicked() {
    downloadFollowers();
    // FIXME cleanup and close dialog
}

function onStopped() {
    dialog.dialog('widget').find('.crawler-download-btn').button('enable');
}

async function onFollowerButtonClicked() {
    var evt = document.createEvent("MouseEvents");
    evt.initEvent('click', true, true);
    authorId = __initialData.data.entry_data.ProfilePage[0].graphql.user.id;
    $('a[href$="/followers/"]').get(0).dispatchEvent(evt);
    await sleep(2000);
    dialog.dialog('open');
}

async function scrollToBottom(selector) {
    await sleep(1000);
    const scrollable = $(selector);

    if (scrollable[0].scrollHeight - scrollable[0].scrollTop === scrollable[0].clientHeight) {
        onStopped();
    } else {
        if (!stopped) {
            scrollable.scrollTop(scrollable[0].scrollHeight);
        } else {
            onStopped();
        }
    }
}

function downloadFollowers() {
    const blob = new Blob([JSON.stringify(followers)], { type: 'text/plain;charset=utf-8' });
    saveAs(blob, 'followers_' + authorId + '.json');
}

(function (open) {
    XMLHttpRequest.prototype.open = function (method, url, async, user, pass) {
        if (url.indexOf('/followers/') >= 0) {
            // rewriting url
            url = url.replace(/count=\d+/, `count=${followersPerPage}`)
            // const cursor = 'QVFBbXVwT2tMT2R4aWNNWDdaR0ItbWxORWxmME9lMzRCZVhTZTltQ0tlSFhPTXUwQjM3Zll1b2pWem9TYWx2MkpFOHl4QVpqVmJXNGhCdUhIVkVFYmxDRQ%3D%3D';
            // url = url.replace(new RegExp(`count=${followersPerPage}&search_surface`), `count=${followersPerPage}&max_id=${cursor}&search_surface`);
            this.addEventListener('readystatechange', () => {
                if (this.readyState === 4) {
                    console.log('Response received');
                    const response = JSON.parse(this.responseText);
                    // console.log(response);
                    followers = followers.concat(response.users);

                    // rewriting response
                    response.users = followers.slice(-12);
                    Object.defineProperty(this, 'responseText', {writable: true});
                    this.responseText = JSON.stringify(response);

                    progressBar.progressbar({
                        max: Math.min(__initialData.data.entry_data.ProfilePage[0].graphql.user.edge_followed_by.count, maxFollowersPerAccount),
                        value: followers.length
                    });
                    if (followers.length > maxFollowersPerAccount) {
                        onStopped();
                    } else {
                        scrollToBottom('div[aria-label="Followers"] .isgrP');
                    }
                }
            }, false);
        }
        open.call(this, method, url, async, user, pass);
    };
})(XMLHttpRequest.prototype.open);

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

(function () {
    'use strict';

    waitForKeyElements('article', init, true);

})();
